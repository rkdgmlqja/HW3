# HW3

## RNN과 LSTM의 Loss 그래프 분석
![train_val_losses](https://github.com/rkdgmlqja/HW3/assets/33273567/cc437b7d-bbc0-4181-9184-78bc0baedeab)
위 그래프를에 따르면 RNN의 Training 및 Validation Loss의 감소 속도가 LSTM에 비해 가파른 모습을 보이며 따라서 over fitting도 이르게 발생하는 모습을 보인다. 


Training Loss의 경우 LSTM Loss감소 기울기가 완만하나 최종적으로는 RNN보다 Training Loss가 낮아지는 모습을 보인다.


Validation Loss의 경우 RNN은 23 Epoch에서 최소값을 가지며 이후에는 다시 over fittiing으로 인하여 증가하고 80epoch이후에는 안정화되는 모습을 보인다.


마찬가지로 LSTM도 38epoch에서 Validation Loss최소값을 가지며 이후에는 다시 증가하는 모습을 보인다.


결과적으로 더 이른 epoch에서 더 낮은 Validation Loss가 낮은 RNN이 최종적으로 Language Generation Task성능이 좋다고 볼 수 있다.



## temperature parameter T에 따른 generation결과 분석

### tempereture: 0.1

### tempereture: 0.3

### tempereture: 0.5

### tempereture: 0.8

### tempereture: 1.0
