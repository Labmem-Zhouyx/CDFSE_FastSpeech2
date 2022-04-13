export LANG="en_US.UTF-8"
python3 synthesize.py \
 	--source synsamples/sents_large.txt \
 	--restore_step 250000 \
 	--mode batch \
 	-p /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3_spkcls/preprocess.yaml \
 	-m /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3_spkcls/model.yaml \
 	-t /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3_spkcls/train.yaml
#python3 synthesize.py \
#	--text "特工特工特工特工特工." \
#	--restore_step 500000 \
#	--mode single \
#	-p /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3_nocontent/preprocess.yaml \
#	-m /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3_nocontent/model.yaml \
#	-t /apdcephfs/share_1316500/yatsenzhou/configs/FS2_CDFSE/AISHELL3_nocontent/train.yaml \
#	--ref_mel /data/training_data/preprocessed_data/AISHELL3/mel/SSB0710-mel-SSB07100001.npy 

