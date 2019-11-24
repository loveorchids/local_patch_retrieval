cd ~/Documents/local_patch_retrieval/

# 191114
#python3 lp_retrieval.py --feature_comb conv --loss_weight l1 --epoch_num 400 --model_prefix lp_rtv_conv_l1 --batch_size_per_gpu 4 --normalize_embedding
#python3 lp_retrieval.py --feature_comb conv --loss_weight l2 --epoch_num 400 --model_prefix lp_rtv_conv_l2 --batch_size_per_gpu 4 --normalize_embedding
#python3 lp_retrieval.py --feature_comb conv --loss_weight default --epoch_num 400 --model_prefix lp_rtv_conv_kl --batch_size_per_gpu 4 --normalize_embedding

# 191115
python3 lp_retrieval.py --feature_comb conv --loss_weight default --epoch_num 3000 --model_prefix lp_rtv_conv_l1 --batch_size_per_gpu 4 --global_embedding
#python3 lp_retrieval.py --feature_comb conv --loss_weight l1 --epoch_num 100 --model_prefix lp_rtv_conv_l1 --batch_size_per_gpu 4
#python3 lp_retrieval.py --feature_comb conv --loss_weight l2 --epoch_num 100 --model_prefix lp_rtv_conv_l2 --batch_size_per_gpu 4