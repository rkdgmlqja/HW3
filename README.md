# HW3

## RNN과 LSTM의 Loss 그래프 분석
![train_val_losses](https://github.com/rkdgmlqja/HW3/assets/33273567/cc437b7d-bbc0-4181-9184-78bc0baedeab)
위 그래프를에 따르면 RNN의 Training 및 Validation Loss의 감소 속도가 LSTM에 비해 가파른 모습을 보이며 따라서 over fitting도 이르게 발생하는 모습을 보인다. 


Training Loss의 경우 LSTM Loss감소 기울기가 완만하나 최종적으로는 RNN보다 Training Loss가 낮아지는 모습을 보인다.


Validation Loss의 경우 RNN은 23 Epoch에서 최소값을 가지며 이후에는 다시 over fittiing으로 인하여 증가하고 80epoch이후에는 안정화되는 모습을 보인다.


마찬가지로 LSTM도 38epoch에서 Validation Loss최소값을 가지며 이후에는 다시 증가하는 모습을 보인다.


결과적으로 더 이른 epoch에서 더 낮은 Validation Loss가 낮은 RNN이 최종적으로 Language Generation Task성능이 좋다고 볼 수 있다.



## temperature parameter T에 따른 generation결과 분석

### temperature: 0.1 Results
**seed1**

MENENIUS:
I will not say the such a soul of the people, come, my lord, and with the commons and ca

**seed2**

CORIOLANUS:
The comes the country's country to the country that have been you have been you to the

**seed3**

SICINIUS:
The country that have not the people with the people, when the country the common the pe

**seed4**

CORIOLANUS:
I will not should be the people, who come to the people to the state the common world 

**seed5**

CORIOLANUS:
The commons and content of the people, come to the commons, and with the consul, that 

### temperature: 0.3 Results
**seed1**

Second Citizen:
Nay, but speak not maliciously.

**seed2**

CORIOLANUS:
What me well, we have not the people comes the consul, we comes the state the city of 

**seed3**

MARCIUS:
The comes to come to the may they do you could shall be the gods of this pass'd, and the

**seed4**

CORIOLANUS:
I will not work with his country's country's death, and made the state a country's bea

**seed5**

CORIOLANUS:
I will see the king, and made the such a charge his country's part of the truth the go

### temperature: 0.5 Results
**seed1**

Second Citizen:
You shall be consul, the wars to make them to me, and the man I will make my son a

**seed2**

BUCKINGHAM:
What, no, not so it forth the people, and for this proud to be a shame, and you shall 

**seed3**

GLOUCESTER:
Why, the man shall not for that that I shall tell his house, that with this dearer and

**seed4**

When set down this a man stand against the wounds and and you shall not shall be consul. I would be

**seed5**

CLARENCE:
O, now this lords, and we do but have men and his worthy wife a good state,
And cry and 

### temperature: 0.8 Results
**seed1**

The combastomed in pride of our power, you would may charge out, and be brain.

First Citizen:
Fare

**seed2**

Secandance:
Give it was belove to to unto the noble to that you way ever be was as you intern'd to

**seed3**

LADY ANNE:
Nabk, when his grace shall be speak; all that high place in holy ready made the part mi

**seed4**

I'll may go't, he shall we shall be at you well under a couting that we wish we in the custens him.

**seed5**

CORIOLANUS:
O was a bloody you have the other need's unless. He me he will.

First Citizen:
Doth a
### temperature: 1.0 Results
**seed1**

SICINIUS:
The corriam's, my lord?

CORIOLANUS:
So the belly sweet way we is name, 

**seed2**

AEdile:
The his more.

SICINIUS:
Where's is true, look'd old may be at they latible qued him that 

**seed3**

YORET:
Grandam tends upon afface, and set thee.

CORIOLANUS:
Whreak'st, you canst from him; you, w

**seed4**

COMINIUS:
O, that you tribune doup's own cruel been by you? my body; and your lay you in catacles 


**seed5**

Second Senator:
But if we show't, who have war as treeth cames? The compation's line.

YORK:
Have 
