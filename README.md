# 安装  

1.创建conda环境
```python
  conda create -n pcam python=3.7
  conda activate pcam  
```

2.安装依赖
```python
  cd pcam/
  pip install -r requirements.txt -i https://pypi.doubanio.com/simple/
  conda install pytorch==1.8.1 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
  conda install pytorch3d=0.6.2
  pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
  pip install -e ./lcp
  pip install -e ./
```

3.运行
```python
  #训练
  python /home/dingfei/projects/hush-PCAM/pcam/scripts/train.py
  #测试
  python /home/dingfei/projects/hush-PCAM/pcam/scripts/eval.py
```
