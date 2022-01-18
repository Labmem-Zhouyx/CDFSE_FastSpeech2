export LANG="en_US.UTF-8"
# python3 synthesize.py \
# 	--source synsamples/sents_redemo.txt \
# 	--restore_step 100000 \
# 	--mode batch \
# 	-p /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3/preprocess.yaml \
# 	-m /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3/model.yaml \
# 	-t /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3/train.yaml
python3 synthesize.py \
	--text "醉后不知天在水,满船清梦压星河." \
	--restore_step 50000 \
	--mode single \
	-p /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3/preprocess.yaml \
	-m /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3/model.yaml \
	-t /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3/train.yaml \
	--ref_mel /data/training_data/preprocessed_data/AISHELL3/mel/SSB0005-mel-SSB00050001.npy 

