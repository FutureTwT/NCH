import torch.nn.functional as F
import torch.nn as nn
import scipy.io as sio

from data_loader import *
import model
import utils


class GCIC(object):
    def __init__(self, config, logger, running_cnt):

        self.running_cnt = running_cnt

        self.alpha_train = config.alpha_train
        self.beta_train = config.beta_train
        self.alpha_query = config.alpha_query
        self.beta_query = config.beta_query
        self.dataset = config.dataset

        complete_data, train_missed_data, query_missed_data = load_data(self.dataset, self.alpha_train, self.beta_train, self.alpha_query, self.beta_query)

        self.config = config
        self.logger = logger

        self.dropout = config.dropout
        self.EPOCHS = config.epochs
        self.WU_EPOCHS = config.warmup_epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.nbit = config.nbit
        self.image_hidden_dim = config.image_hidden_dim
        self.text_hidden_dim = config.text_hidden_dim
        self.fusion_dim = config.fusion_dim

        self.ANCHOR = config.ANCHOR

        ########################################### data ###########################################
        self.train_data = [complete_data['I_tr'], complete_data['T_tr']]
        self.train_labels = complete_data['L_tr'].numpy()

        self.retrieval_data = [complete_data['I_db'], complete_data['T_db']]
        self.retrieval_labels = complete_data['L_db'].numpy()

        # train missed data
        self.train_dual_data = [train_missed_data['I_dual_img'], train_missed_data['I_dual_txt']]
        self.train_dual_labels = train_missed_data['I_dual_label']

        self.train_only_imgs = train_missed_data['I_oimg'] 
        self.train_only_imgs_labels = train_missed_data['I_oimg_label'] 

        self.train_only_txts = train_missed_data['I_otxt']  
        self.train_only_txts_labels = train_missed_data['I_otxt_label'] 

        # query missed data
        self.query_dual_data = [query_missed_data['I_dual_img'], query_missed_data['I_dual_txt']]
        self.query_only_imgs = query_missed_data['I_oimg'] 
        self.query_only_txts = query_missed_data['I_otxt']  

        self.query_labels = torch.cat((query_missed_data['I_dual_label'], query_missed_data['I_oimg_label'], query_missed_data['I_otxt_label'])).numpy()

        self.train_nums = self.train_labels.shape[0]
        self.train_dual_nums = self.train_dual_data[0].size(0)
        self.train_only_imgs_nums = self.train_only_imgs.size(0)
        self.train_only_txts_nums = self.train_only_txts.size(0)
        assert self.train_nums == (self.train_dual_nums + self.train_only_imgs_nums + self.train_only_txts_nums)

        self.batch_dual_size = math.ceil(self.batch_size * (1 - self.alpha_train))
        self.batch_img_size = math.floor(self.batch_size * self.alpha_train * self.beta_train)
        self.batch_txt_size = self.batch_size - self.batch_dual_size - self.batch_img_size
        assert self.batch_txt_size >= 0

        self.img_dim = self.train_data[0].size(1)
        self.txt_dim = self.train_data[1].size(1)
        self.num_classes = self.train_labels.shape[1]
        self.logger.info('Dataset-%s has %d classes!' % (self.dataset, self.num_classes))

        #################################### model define ##########################################
        # specific hash function
        self.img_mlp_enc = model.MLP(units=[self.img_dim, self.image_hidden_dim, self.fusion_dim])
        self.txt_mlp_enc = model.MLP(units=[self.txt_dim, self.text_hidden_dim, self.fusion_dim])
        # TEs for obtain neighbour information
        self.img_TEs_enc = model.TransformerEncoder(Q_dim=self.txt_dim, K_dim=self.txt_dim, V_dim=self.img_dim)
        self.txt_TEs_enc = model.TransformerEncoder(Q_dim=self.img_dim, K_dim=self.img_dim, V_dim=self.txt_dim)
        # missing data generator
        self.img_ffn_enc = model.FFNGenerator(input_dim=self.txt_dim, output_dim=self.img_dim)
        self.txt_ffn_enc = model.FFNGenerator(input_dim=self.img_dim, output_dim=self.txt_dim)
        # final hash function
        self.fusion_model = model.Fusion(fusion_dim=self.fusion_dim, nbit=self.nbit)

        if torch.cuda.is_available():
            self.img_mlp_enc.cuda(), self.txt_mlp_enc.cuda()
            self.img_TEs_enc.cuda(), self.txt_TEs_enc.cuda()
            self.img_ffn_enc.cuda(), self.txt_ffn_enc.cuda()
            self.fusion_model.cuda()

        ################################# criterion define #########################################
        self.reconstruction_criterion = nn.MSELoss()

        ################################# optimizer define #########################################
        self.optimizer = torch.optim.Adam(
            [{"params": self.img_mlp_enc.parameters(), "lr": self.lr},
             {"params": self.txt_mlp_enc.parameters(), "lr": self.lr},
             {"params": self.img_ffn_enc.parameters(), "lr": self.lr},
             {"params": self.txt_ffn_enc.parameters(), "lr": self.lr},
             {"params": self.fusion_model.parameters(), "lr": self.lr}])

        self.TEs_optimizer = torch.optim.Adam(
            [{"params": self.img_TEs_enc.parameters(), "lr": self.lr},
             {"params": self.txt_TEs_enc.parameters(), "lr": self.lr}])

        ################################## anchor extract ##########################################
        self.anchor_nums = config.anchor_nums
        if self.anchor_nums > self.train_dual_nums:
            self.logger.critical('The anchor number is large than the number of dual samples.')
            self.anchor_nums = self.train_dual_nums

        # anchor must belong to train dual data!
        self.anchor_idx = np.random.permutation(self.train_dual_nums)[:self.anchor_nums]

        self.img_anchor = self.train_dual_data[0][self.anchor_idx, :].cuda()
        self.txt_anchor = self.train_dual_data[1][self.anchor_idx, :].cuda()
        self.anchor_label = self.train_dual_labels[self.anchor_idx, :].cuda()

        self.query_code = None
        self.retrieval_code = None
        
        ################################# hyper-parameter define ####################################
        self.param_neighbour = config.param_neighbour
        self.param_sim = config.param_sim
        self.param_sign = config.param_sign

        self.batch_count = int(math.ceil(self.train_nums / self.batch_size))

        ################################# metric value ####################################
        self.average_map = 0
    
    def warmup(self):
        self.img_TEs_enc.train(), self.txt_TEs_enc.train()

        self.train_loader = data.DataLoader(TrainCoupledData(self.train_dual_data[0], self.train_dual_data[1], self.train_dual_labels),
                                                             batch_size=self.batch_size, shuffle=True) 

        for epoch in range(self.WU_EPOCHS):
            for batch_idx, (img_forward, txt_forward, label) in enumerate(self.train_loader):
                img_forward = img_forward.cuda()
                txt_forward = txt_forward.cuda()
                label = label.cuda()
                self.TEs_optimizer.zero_grad()

                graph = utils.GEN_S_GPU(label, self.anchor_label)

                img_neighbour = self.img_TEs_enc(txt_forward, self.txt_anchor, self.img_anchor, graph)
                txt_neighbour = self.txt_TEs_enc(img_forward, self.img_anchor, self.txt_anchor, graph)  

                LOSS = self.reconstruction_criterion(img_neighbour, img_forward) + \
                       self.reconstruction_criterion(txt_neighbour, txt_forward)

                LOSS.backward()
                self.TEs_optimizer.step()

                if batch_idx == 0:
                    self.logger.info('[%4d/%4d] (Warm-up) Loss: %.4f' % (epoch + 1, self.WU_EPOCHS, LOSS.item()))

    def train(self):
        self.img_mlp_enc.train(), self.txt_mlp_enc.train()
        self.img_ffn_enc.train(), self.txt_ffn_enc.train()
        self.img_TEs_enc.train(), self.txt_TEs_enc.train()
        self.fusion_model.train()
        
        dual_idx = np.arange(self.train_dual_nums)
        oimg_idx = np.arange(self.train_only_imgs_nums)
        otxt_idx = np.arange(self.train_only_txts_nums)

        for epoch in range(self.EPOCHS):

            np.random.shuffle(dual_idx)
            np.random.shuffle(oimg_idx)
            np.random.shuffle(otxt_idx)

            for batch_idx in range(self.batch_count):
                small_dual_idx = dual_idx[batch_idx * self.batch_dual_size: (batch_idx + 1) * self.batch_dual_size]
                small_oimg_idx = oimg_idx[batch_idx * self.batch_img_size: (batch_idx + 1) * self.batch_img_size]
                small_otxt_idx = otxt_idx[batch_idx * self.batch_txt_size: (batch_idx + 1) * self.batch_txt_size]

                train_dual_img = self.train_dual_data[0][small_dual_idx, :].cuda()
                train_dual_txt = self.train_dual_data[1][small_dual_idx, :].cuda()
                train_dual_labels = self.train_dual_labels[small_dual_idx, :].cuda()

                train_only_img = self.train_only_imgs[small_oimg_idx, :].cuda()
                train_only_img_labels = self.train_only_imgs_labels[small_oimg_idx, :].cuda()
   
                train_only_txt = self.train_only_txts[small_otxt_idx, :].cuda()
                train_only_txt_labels = self.train_only_txts_labels[small_otxt_idx, :].cuda()

                loss = self.trainstep(train_dual_img, train_dual_txt, train_dual_labels, 
                                      train_only_img, train_only_img_labels, train_only_txt, train_only_txt_labels)

                if (batch_idx + 1) == self.batch_count:
                    self.logger.info('[%4d/%4d] Loss: %.4f' % (epoch + 1, self.EPOCHS, loss))

    def trainstep(self, train_dual_img, train_dual_txt, train_dual_labels,
                  train_only_img, train_only_img_labels, train_only_txt, train_only_txt_labels):

        self.optimizer.zero_grad()

        dual_cnt = train_dual_labels.size(0) 
                                                                             
        img_forward = torch.cat([train_dual_img, train_only_img])
        txt_forward = torch.cat([train_dual_txt, train_only_txt])
        img_labels = torch.cat([train_dual_labels, train_only_img_labels])
        txt_labels = torch.cat([train_dual_labels, train_only_txt_labels])
        labels = torch.cat([train_dual_labels, train_only_img_labels, train_only_txt_labels])

        # construct graph
        img_graph = utils.GEN_S_GPU(img_labels, self.anchor_label)
        txt_graph = utils.GEN_S_GPU(txt_labels, self.anchor_label)
        graph = utils.GEN_S_GPU(labels, labels)

        ##### Forward
        # specific hash code
        img_feat = self.img_mlp_enc(img_forward)
        txt_feat = self.txt_mlp_enc(txt_forward)

        # generator + specific hash code
        img_recons = self.img_ffn_enc(txt_forward)
        txt_recons = self.txt_ffn_enc(img_forward)    
        img_recons_feat = self.img_mlp_enc(img_recons)
        txt_recons_feat = self.txt_mlp_enc(txt_recons)

        # obtain the neighbour information
        with torch.no_grad():
            img_neighbour = self.img_TEs_enc(txt_forward, self.txt_anchor, self.img_anchor, txt_graph)
            txt_neighbour = self.txt_TEs_enc(img_forward, self.img_anchor, self.txt_anchor, img_graph)  

        # get final hash code
        dual_repre = self.fusion_model(img_feat[:dual_cnt], txt_feat[:dual_cnt])
        oimg_repre = self.fusion_model(img_feat[dual_cnt:], txt_recons_feat[dual_cnt:])
        otxt_repre = self.fusion_model(img_recons_feat[dual_cnt:], txt_feat[dual_cnt:])
        
        total_repre = torch.cat([dual_repre, oimg_repre, otxt_repre])
        total_repre_norm = F.normalize(total_repre)
        B = torch.sign(total_repre)

        ##### loss function
        LOSS_sign = self.reconstruction_criterion(total_repre, B)

        LOSS_sim = self.reconstruction_criterion(total_repre_norm.mm(total_repre_norm.T), graph)

        # Rewrite to avoid NaN
        if img_recons.size(0) != 0 and txt_recons.size(0) == 0:
            LOSS_neighbour = self.reconstruction_criterion(img_recons, img_neighbour)
        elif img_recons.size(0) == 0 and txt_recons.size(0) != 0:
            LOSS_neighbour = self.reconstruction_criterion(txt_recons, txt_neighbour)
        elif img_recons.size(0) != 0 and txt_recons.size(0) != 0:
            LOSS_neighbour = self.reconstruction_criterion(img_recons, img_neighbour) + \
                             self.reconstruction_criterion(txt_recons, txt_neighbour)
        else:
            LOSS_neighbour = None

        LOSS = LOSS_sign * self.param_sign + LOSS_sim * self.param_sim

        if LOSS_neighbour != None:
            LOSS = LOSS + LOSS_neighbour * self.param_neighbour 

        LOSS.backward()
        self.optimizer.step()
        
        return LOSS.item()

    def test(self):
        self.logger.info('[TEST STAGE]')
        self.img_mlp_enc.eval(), self.txt_mlp_enc.eval()
        self.img_ffn_enc.eval(), self.txt_ffn_enc.eval()
        self.fusion_model.eval()
        
        self.logger.info('Retrieval Begin.')
        # retrieval set
        self.retrieval_loader = data.DataLoader(CoupledData(self.retrieval_data[0], self.retrieval_data[1]),
                                                batch_size=2048, shuffle=False) # Dont shuffle!

        retrievalP = []
        for i, (img_forward, txt_forward) in enumerate(self.retrieval_loader):
            img_forward = img_forward.cuda()
            txt_forward = txt_forward.cuda()
            with torch.no_grad():
                img_feat = self.img_mlp_enc(img_forward)
                txt_feat = self.txt_mlp_enc(txt_forward)
                H = self.fusion_model(img_feat, txt_feat)
            retrievalP.append(H.data.cpu().numpy())

        retrievalH = np.concatenate(retrievalP)
        self.retrieval_code = np.sign(retrievalH)
        self.logger.info('Retrieval End.')

        self.logger.info('Query Begin.')
        # query set
        queryP = []
        with torch.no_grad():
            dual_img_feat = self.img_mlp_enc(self.query_dual_data[0].cuda())
            dual_txt_feat = self.txt_mlp_enc(self.query_dual_data[1].cuda())
            dualH = self.fusion_model(dual_img_feat, dual_txt_feat)
        queryP.append(dualH.data.cpu().numpy())

        with torch.no_grad():
            oimg_feat = self.img_mlp_enc(self.query_only_imgs.cuda()) 
            oimg_Gtxt = self.txt_ffn_enc(self.query_only_imgs.cuda())  
            oimg_Gtxt = self.txt_mlp_enc(oimg_Gtxt)  
            oimgH = self.fusion_model(oimg_feat, oimg_Gtxt)
        queryP.append(oimgH.data.cpu().numpy())

        with torch.no_grad():
            otxt_Gimg = self.img_ffn_enc(self.query_only_txts.cuda())
            otxt_Gimg = self.img_mlp_enc(otxt_Gimg)
            otxt_feat = self.txt_mlp_enc(self.query_only_txts.cuda()) 

            otxtH = self.fusion_model(otxt_Gimg, otxt_feat)
        queryP.append(otxtH.data.cpu().numpy())

        queryH = np.concatenate(queryP)
        self.query_code = np.sign(queryH)
        self.logger.info('Query End.')
        
        assert self.retrieval_code.shape[0] == self.retrieval_labels.shape[0]
        assert self.query_code.shape[0] == self.query_labels.shape[0]

        _dict = {
            'retrieval_B': self.retrieval_code.astype(np.int8),
            'query_B': self.query_code.astype(np.int8),
            'cateTrainTest': np.sign(self.retrieval_labels @ self.query_labels.T).astype(np.int8)
        }

        save_path = 'Hashcode/NCH-Ours_%s_%d_%.1f_%.1f_%d.mat' % (self.dataset, self.nbit, self.alpha_train, self.alpha_query, self.running_cnt)
        sio.savemat(save_path, _dict)
        self.logger.info('[Saved]')
