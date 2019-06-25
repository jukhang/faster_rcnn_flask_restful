## Faster Rcnn flask Web RestFul App

#### Faster Rcnn - https://github.com/endernewton/tf-faster-rcnn

#### System && Python

Python3.6 tensorflow1.9 Ubuntu16.04

#### post参数说明

传入图片文件(上线可采用图片云存储，传图片url等)

#### 返回值

```json
{
  "things": [
    [
      {
        "class_name": "bus",
        "position": "[ 62.390697  36.834557 387.46948  233.50587 ]",
        "score": "0.996106"
      }
    ]
  ]
}
```