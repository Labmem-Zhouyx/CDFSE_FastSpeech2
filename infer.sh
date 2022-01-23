export LANG="en_US.UTF-8"
# python3 synthesize.py \
# 	--source synsamples/sents_redemo.txt \
# 	--restore_step 100000 \
# 	--mode batch \
# 	-p /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3/preprocess.yaml \
# 	-m /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3/model.yaml \
# 	-t /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3/train.yaml
python3 synthesize.py \
	--text "白云出演的电视剧有什么." \
	--restore_step 250000 \
	--mode single \
	-p /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3/preprocess.yaml \
	-m /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3/model.yaml \
	-t /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3/train.yaml \
	--ref_mel /data/training_data/preprocessed_data/AISHELL3/mel/SSB0710-mel-SSB07100001.npy 

