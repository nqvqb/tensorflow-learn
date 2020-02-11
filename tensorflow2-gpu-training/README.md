
# tensorflow2-gpu-training

##### Dependencies
```sh
virtualenv --system-site-packages -p python3 ~/venv/tf21
source ~/venv/tf21/bin/activate

pip3 install tensorflow-gpu==2.1

pip3 install pydot
sudo apt-get install graphviz

pip3 install cifar2png
cd ~/datasets
# download cifar-10 dataaset and convert
cifar2png cifar10 cifar-10

pip3 install scipy
pip3 install matplotlib
```


##### train
```sh
tmux new -s train1

python3 1-test_resnet18.py

tmux attach-session -t train1
```

##### trained result
```sh
cifar-10 full test 5
782/782 [==============================] - 59s 75ms/step - loss: 1.8536 - sparse_categorical_accuracy: 0.4754
- val_loss: 3.3374 - val_sparse_categorical_accuracy: 0.2418

```
