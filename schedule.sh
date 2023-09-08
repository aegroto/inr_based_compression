CONFIG_ID=KODAK_1x_epochs25000_lr0.0005_ffdims_16_hdims32_hlayer3_nerf_sine_l1_reg1e-05_enc_scale1.4
EXP_ROOT=exp/basic_kodak/$CONFIG_ID
IMG_ID=kodim01
OUT_FOLDER=results/kodak/$CONFIG_ID/$IMG_ID

mkdir -p $OUT_FOLDER

python3 image_compression/test.py \
    --image_path data/KODAK/$IMG_ID.png \
    --flags_file $EXP_ROOT/FLAGS.yml \
    --exp_folder $EXP_ROOT/$IMG_ID/ \
    --bitwidth 8 \
    --out_folder $OUT_FOLDER
