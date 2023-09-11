EXP_ID=$1
CONFIG_ID=$2
IMG_ID=$3
DATASET_ID=$4
EXP_ROOT=exp/$EXP_ID/$CONFIG_ID
OUT_FOLDER=results/bw$BITWIDTH/$EXP_ID/$CONFIG_ID/$IMG_ID

mkdir -p $OUT_FOLDER

python3 image_compression/test.py \
    --image_path data/$DATASET_ID/$IMG_ID.png \
    --flags_file $EXP_ROOT/FLAGS.yml \
    --exp_folder $EXP_ROOT/$IMG_ID/ \
    --bitwidth $BITWIDTH \
    --out_folder $OUT_FOLDER
