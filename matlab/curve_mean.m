function [] = curve_mean_abs(ABS_name, nbit, db_name, per, per_query, flag)
    addpath('utils/');

    nround = 5; % TODO: mean result!
    mapN_CNT = zeros(nround, 1);
    for n=1:nround
        fprintf('Load hashcode from Our-model.\n');
        save_path = sprintf('../Hashcode/%s_%s_%d_%.1f_%.1f_%d.mat', ABS_name, db_name, nbit, per, per_query, n);
        load(save_path);

        %% compact hashcode
        B_trn = compactbit(retrieval_B > 0);
        B_tst = compactbit(query_B > 0);

        Dhamm = hammingDist(B_tst, B_trn)';

        %% TWT evaluate
        [~, Dhamm_index] = sort(Dhamm, 1);
        [Pre, Rec, MAP] = fast_PR_MAP(int32(cateTrainTest), int32(Dhamm_index));
        mapN_CNT(n, 1) = MAP(end);
    end

    fprintf('[%s-%s][Train-PDR=%.1f / Query-PDR=%.1f] MAP = %.4f (%.4f)\n', db_name, num2str(nbit), per, per_query, mean(mapN_CNT), std(mapN_CNT));

    name = ['../Result/' db_name '_' flag '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%s][Train-PDR=%.1f / Query-PDR=%.1f] MAP = %.4f (%.4f)\n', db_name, num2str(nbit), per, per_query, mean(mapN_CNT), std(mapN_CNT));
end
