# text-classify
a small exercise

## train
run \n
python train.py --train_file YOUR_TRAIN_FILE_PATH --val_file YOUR_VAL_FILE_PATH --num_epochs NUM_EPOCHS --batch_size BATCH_sIZE --lr LEARNING_RATE \n
for example:  \n
python train.py --train_file ./train.tsv --val_file ./test.tsv --num_epochs 50 --batch_size 128 --lr 0.001 \n

## predict
run \n
python predict.py --file_path YOUR_PREDICT_FILE_PATH --model_path YOUR_MODEL_PATH --output_dir OUTPUT_DIRCTORY_PATH   \n
for example: \n
python predict.py --file_path ./small_test.tsv --model_path D:\CC\
数据集\checkpoint\model_epoch50_ValLoss1.82574892.pth --output_dir ./predict   \n

