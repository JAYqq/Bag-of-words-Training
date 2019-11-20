# Bag-of-words-Training

采用词袋模型对图片进行相似度筛选

### 依赖环境

opencv3.3.1

opencv_contrib-3.3.1

### 配置流程

1. 创建虚拟环境

2. ```
   pip install -r requirements.txt
   ```

### 使用

```
#模型训练
python test.py --imgpath <Your ImageList Path>

#测试样本图像
python search.py --imgpath <Your ImageList Path>
```

