case $1 in
  build|train|inference|cluster)
    echo "Valid argument: $1"
    ;;
  *)
    echo "Invalid argument: $1. Please enter correct choice."
    exit 1
    ;;
esac


if [ "$1" = "build" ]; then
    python src/MILP_utils.py --mode=model2data --input_dir=dataset/model --output_dir=dataset/data --type=direct
fi

if [ "$1" = "train" ]; then
    python src/train.py --dataset=MILP --cfg_path=experiments/configs/test.yml --seed=1 --device=0 --model_path=experiments/weights/encoder.pth --dataset_path=dataset/data || echo "Training failed"
fi

if [ "$1" = "inference" ]; then
    python src/inference.py --cfg_path=experiments/configs/test.yml --seed=1 --device=0 --model_path=experiments/weights/encoder.pth --input_dir=dataset/data --output_file=tmp.pkl --filename=namelist.pkl || echo "Inference failed"
fi

if [ "$1" = "cluster" ]; then
    python src/clustering.py --filename=namelist.pkl --input_file=tmp.pkl
fi
