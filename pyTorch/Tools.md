# pyTorch Tools 

## Visdom for PyTorch Visualization
```bash
pip install visdom
python -m visdom.server
```
Open `http://localhost:8097`

> 출처 : [Visdom github](https://github.com/facebookresearch/visdom)

###### [Error] unsupported GNU version! gcc versions later than 5 are not supported!
```bash 
sudo apt-get install python-software-properties
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-4.9
sudo apt-get install g++-4.9

sudo rm /usr/bin/g++
sudo ln -s /usr/bin/g++-4.9 /usr/bin/g++

sudo rm /usr/bin/gcc
sudo ln -s /usr/bin/gcc-4.9 /usr/bin/gcc




---

## Tensorboard 

[tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger): 텐서플로우없이 텐서보드만 사용하는법 

[Crayon](https://github.com/torrvision/crayon): framework that gives you access to the visualisation power of TensorBoard with python

[tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch): Write tensorboard events with simple command.
