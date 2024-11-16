+++
title = 'DeepTest 2'
date = 2024-07-31T17:45:38+09:00
draft = false
+++

이 글은 DeepTest: Automated Testing of Deep-Neural-Network-driven Autonomous Cars (https://arxiv.org/abs/1708.08559)을 번역 및 요약한 글입니다.
[DeepTest 1]({{< ref "DeepTest-1" >}}) 에서 이어집니다.

## 5. RESULTS
As DNN-based models are fundamentally different than traditional software, first, we check whether neuron coverage is a good metric to capture functional diversity of DNNs.
In particular, we investigate whether neuron coverage changes with different input-output pairs of an autonomous car.
An individual neuron’s output goes through a sequence of linear and nonlinear operations before contributing to the final outputs of a DNN.
Therefore, it is not very clear how much (if at all) individual neuron’s activation will change the final output.
We address this in our first research question.

딥러닝 기반 모델(DNN)은 전통적인 소프트웨어와 근본적으로 다르기 때문에, 먼저 뉴런 커버리지가 DNN의 기능적 다양성을 포착하는 좋은 지표인지 확인합니다.
특히, 자율 주행 자동차의 다양한 입력-출력 쌍에 따라 뉴런 커버리지가 어떻게 변하는지 조사합니다.
개별 뉴런의 출력은 최종 출력에 기여하기 전에 일련의 선형 및 비선형 연산을 거치므로, 개별 뉴런의 활성화가 최종 출력에 얼마나 영향을 미칠지는 명확하지 않습니다.
우리는 첫 번째 연구 질문에서 이를 다룹니다.

### RQ1. Do different input-output pairs result in different neuron coverage?
As steering angle is a continuous variable, we check Spearman rank correlation [76] between neuron coverage and steering angle.
Table 5 shows that Spearman correlations for all the models are statistically significant—while Chauffeur and Rambo models show an overall negative association, Epoch model shows a strong positive correlation.
This result indicates that the neuron coverage changes with the changes in output steering angles, i.e. different neurons get activated for different outputs.
Thus, in this setting, neuron coverage can be a good approximation for estimating the diversity of input-output pairs.

조향 각도는 연속 변수이므로 뉴런 커버리지와 조향 각도 간의 스피어만 순위 상관 계수를 확인합니다[76].
표 5는 모든 모델에 대해 스피어만 상관 계수가 통계적으로 유의미함을 보여줍니다.
Chauffeur와 Rambo 모델은 전반적으로 부정적인 상관 관계를 보이는 반면, Epoch 모델은 강한 양의 상관 관계를 보입니다.
이 결과는 출력 조향 각도의 변화에 따라 뉴런 커버리지가 변한다는 것을 나타내며, 즉, 다른 출력에 대해 다른 뉴런이 활성화된다는 것을 의미합니다.
따라서 이러한 설정에서 뉴런 커버리지는 입력-출력 쌍의 다양성을 추정하는 데 좋은 근사치가 될 수 있습니다.

![Table 5](/posts/paper/DeepTest/table5.png)

To measure the association between neuron coverage and steering direction, we check whether the coverage varies between right and left steering direction.
We use the Wilcoxon nonparametric test as the steering direction can only have two values (left and right).
Our results confirm that neuron coverage varies with steering direction with statistical significance (p < 2.2 ∗ 10^−16) for all the three overall models.
Interestingly, for Rambo , only the Rambo-S1 sub-model shows statistically significant correlation but not Rambo-S2 and Rambo-S3. 

뉴런 커버리지와 조향 방향 간의 연관성을 측정하기 위해, 커버리지가 좌회전과 우회전 방향에 따라 달라지는지 확인합니다.
조향 방향은 좌회전과 우회전 두 값만 가질 수 있으므로, 비모수적 테스트인 Wilcoxon 검정을 사용합니다.
우리의 결과는 세 가지 전체 모델 모두에 대해 뉴런 커버리지가 조향 방향에 따라 통계적으로 유의미하게(p < 2.2 * 10^-16) 변한다는 것을 확인합니다.
흥미롭게도 Rambo 모델의 경우, Rambo-S1 하위 모델만이 통계적으로 유의미한 상관 관계를 보였으며, Rambo-S2와 Rambo-S3는 그렇지 않았습니다.

Overall, these results show that neuron coverage altogether varies significantly for different input-output pairs. Thus, a neuron-coverage-directed (NDG) testing strategy can help in finding corner cases.

전체적으로 이러한 결과는 다양한 입력-출력 쌍에 대해 뉴런 커버리지가 상당히 달라진다는 것을 보여줍니다. 따라서 뉴런 커버리지 기반(NDG) 테스트 전략이 코너 케이스를 찾는 데 도움이 될 수 있습니다.

**Result 1**: 뉴런 커버리지는 입력-출력 다양성과 상관관계가 있으며, 체계적인 테스트 생성에 사용할 수 있습니다.

### RQ2. Do different realistic image transformations activate different neurons?
We randomly pick 1,000 input images from the test set and transform each of them by using seven different transformations: blur, brightness, contrast, rotation, scale, shear, and translation.
We also vary the parameters of each transformation and generate a total of 70,000 new synthetic images.
We run all models with these synthetic images as input and record the neurons activated by each input.

우리는 테스트 세트에서 임의로 1,000개의 입력 이미지를 선택하고, 각 이미지를 7가지 다른 변환(블러, 밝기, 대비, 회전, 크기, 기울임, 평행 이동)으로 변환합니다.
또한 각 변환의 파라미터를 다양하게 조정하여 총 70,000개의 새로운 합성 이미지를 생성합니다.
이러한 합성 이미지를 입력으로 사용하여 모든 모델을 실행하고, 각 입력에 의해 활성화된 뉴런을 기록합니다.

We then compare the neurons activated by different synthetic images generated from the same image.
Let us assume that two transformations T1 and T2 , when applied to an original image I , activate two sets of neurons N1 and N1 respectively.
We measure the dissimilarities between N1 and N2 by measuring their Jaccard
distance: \[ 1 - \frac{|N_1 \cap N_2|}{|N_1 \cup N_2|} \]

그런 다음 우리는 동일한 원본 이미지에서 생성된 다양한 합성 이미지에 의해 활성화된 뉴런을 비교합니다.
원본 이미지 \(I\)에 두 가지 변환 \(T1\)과 \(T2\)를 적용했을 때, 각각의 변환이 두 개의 뉴런 집합 \(N1\)과 \(N2\)를 활성화한다고 가정해 봅시다.
우리는 \(N1\)과 \(N2\) 간의 유사성을 Jaccard 거리로 측정합니다.

Figure 4.1 shows the result for all possible pair of transformations (e.g., blur vs. rotation, rotation vs. transformation, etc.) for different models.
These results indicate that for all models, except ChauffeurLSTM , different transformations activate different neurons.
As discussed in Section 2.1, LSTM is a particular type of RNN architecture that keeps states from previous inputs and hence increasing the neuron coverage of LSTM models with single transformations is much harder than other models.

그림 4.1은 다양한 모델에 대해 모든 가능한 변환 쌍(예: 블러 대 회전, 회전 대 변환 등)의 결과를 보여줍니다.
이러한 결과는 Chauffeur-LSTM을 제외한 모든 모델에서 다양한 변환이 서로 다른 뉴런을 활성화한다는 것을 나타냅니다.
섹션 2.1에서 논의된 바와 같이, LSTM은 이전 입력의 상태를 유지하는 특정 유형의 RNN 아키텍처이므로 단일 변환으로 LSTM 모델의 뉴런 커버리지를 증가시키는 것이 다른 모델보다 훨씬 더 어렵습니다. 

![Figure 4.1](/posts/paper/DeepTest/figure4_1.png)

We further check how much a single transformation contributes in increasing the neuron coverage w.r.t. all other transformations for a given seed image.
Thus, if an original image I undergoes seven discrete transformations: T1,T2, ...T7, we compute the total number 7 of neurons activated by \( T1, T1 \cup T2, \ldots, \bigcup_{i=1}^{7} Ti \).
Figure 4.2 shows the i=1 cumulative effect of all the transformations on average neuron coverage per seed image.
We see that the cumulative coverage increases with increasing number of transformations for all the models.
In other words, all the transformations are contributing towards the overall neuron coverage.

우리는 또한 단일 변환이 주어진 시드 이미지에 대해 다른 모든 변환에 비해 뉴런 커버리지를 얼마나 증가시키는지 확인합니다.
따라서 원본 이미지 \( I \)가 7개의 개별 변환 \( T1, T2, \ldots, T7 \)을 겪는 경우, \( T1, T1 \cup T2, \ldots, \bigcup_{i=1}^{7} Ti \)에 의해 활성화된 뉴런의 총 수를 계산합니다.
그림 4.2는 시드 이미지당 평균 뉴런 커버리지에 대한 모든 변환의 누적 효과를 보여줍니다.
모든 모델에서 변환의 수가 증가함에 따라 누적 커버리지가 증가하는 것을 볼 수 있습니다.
다시 말해, 모든 변환이 전체 뉴런 커버리지에 기여하고 있음을 알 수 있습니다.

![Figure 4.2](/posts/paper/DeepTest/figure4_2.png)

We also compute the percentage change in neuron coverage per image transformation (NT ) w.r.t. to the corresponding seed image (NO ) as: (NT -NO )/NO .
Figure 5 shows the result.
For all the studied models, the transformed images increase the neuron coverage significantly—Wilcoxon nonparametric test confirms the statistical significance. 

우리는 또한 각 이미지 변환(NT)에 대한 해당 시드 이미지(NO) 대비 뉴런 커버리지의 백분율 변화를 다음과 같이 계산합니다: (NT - NO) / NO.
그림 5는 그 결과를 보여줍니다.
연구된 모든 모델에서 변환된 이미지가 뉴런 커버리지를 상당히 증가시킴을 보여줍니다.
Wilcoxon 비모수 검정은 통계적 유의성을 확인해줍니다.

![Figure 5](/posts/paper/DeepTest/figure5.png)

**Result 2**: 다양한 이미지 변환은 서로 다른 뉴런 집합을 활성화하는 경향이 있습니다.

### RQ3. Can neuron coverage be further increased by combining different image transformations?
We perform this experiment by measuring neuron coverage in two different settings: (i) applying a set of transformations and (ii) combining transformations using coverage-guided search.

우리는 두 가지 다른 설정에서 뉴런 커버리지를 측정하여 이 실험을 수행합니다: (i) 일련의 변환 적용 및 (ii) 커버리지 유도 검색을 사용한 변환 조합.

i) **Cumulative Transformations**
we generate a total of 7,000 images from 100 seed images by applying 7 transformations and varying 10 parameters on 100 seed images.
This results in a total of 7,000 test images.
We then compare the cumulative neuron coverage of these synthetic images w.r.t. the baseline, which use the same 100 seed images for fair comparison.
Table 6 shows the result.

i) **누적 변환** 우리는 100개의 시드 이미지에 7가지 변환을 적용하고 10개의 매개변수를 다양하게 조정하여 총 7,000개의 이미지를 생성합니다.
이로 인해 총 7,000개의 테스트 이미지가 생성됩니다.
그런 다음 동일한 100개의 시드 이미지를 사용하여 공정한 비교를 위해 누적 기준 커버리지를 기준으로 이러한 합성 이미지의 누적 뉴런 커버리지를 비교합니다.
표 6은 그 결과를 보여줍니다.

![Table 6](/posts/paper/DeepTest/table6.png)

ii) **Guided Transformations** Finally, we check whether we can further increase the cumulative neuron coverage by using the coverage-guided search technique described in Algorithm 1.
We generate 254, 221, and 864 images from 100 seed images for Chauffeur-CNN , Epoch , and Rambo models respectively and measure their collective neuron coverage.
As shown in Table 6, the guided transformations collectively achieve 88%, 51%, 64%, 70%, and 98% of total neurons for models Chauffeur-CNN , Epoch , Rambo-S1 , Rambo-S2 , and Rambo-S3 respectively, thus increasing the coverage up to 17% 22%, 12%, 21%, and 0.5% w.r.t. the unguided approach.
This method also significantly achieves higher neuron coverage w.r.t. baseline cumulative coverage.

ii) **유도 변환** 마지막으로, 알고리즘 1에 설명된 커버리지 유도 검색 기법을 사용하여 누적 뉴런 커버리지를 더 증가시킬 수 있는지 확인합니다.
우리는 Chauffeur-CNN, Epoch, 및 Rambo 모델에 대해 각각 254, 221, 및 864개의 이미지를 100개의 시드 이미지에서 생성하고, 이들의 집합적인 뉴런 커버리지를 측정합니다.
표 6에서 보듯이, 유도 변환은 모델 Chauffeur-CNN, Epoch, Rambo-S1, Rambo-S2, 및 Rambo-S3에 대해 각각 전체 뉴런의 88%, 51%, 64%, 70%, 및 98%를 달성하여 비유도 접근법 대비 커버리지를 각각 최대 17%, 22%, 12%, 21%, 및 0.5% 증가시켰습니다.
이 방법은 또한 기준 누적 커버리지 대비 뉴런 커버리지를 상당히 높이는 데 성공했습니다.

**결과 3**: 다양한 이미지 변환을 체계적으로 결합함으로써, 원본 시드 이미지로 달성한 커버리지에 비해 뉴런 커버리지를 약 100% 향상시킬 수 있습니다.

### RQ4. Can we automatically detect erroneous behaviors using metamorphic relations?
We comparethedeviationbetweentheoutputsofIerr andIorд w.r.t.the corresponding human labels, as shown in Figure 6.
The deviations produced for Ierr are much larger than Iorд (also confirmed by Wilcoxon test for statistical significance).
In fact, mean squared error (MSE) for I is 0.41, while the MSE of the corresponding err Iorд is 0.035.
Such differences also exist for other synthetic images that are generated by composite transformations including rain, fog, and those generated during the coverage-guided search.
Thus, overall Ierr has a higher potential to show buggy behavior.

그림 6에서 보듯이, 우리는 \( I_{err} \)와 \( I_{org} \)의 출력 간의 편차를 해당 인간 레이블과 비교합니다.
\( I_{err} \)에 대해 생성된 편차는 \( I_{org} \)에 비해 훨씬 크며, 이는 통계적 유의성을 위한 Wilcoxon 테스트에서도 확인되었습니다.
실제로 \( I_{err} \)의 평균 제곱 오차(MSE)는 0.41인 반면, 해당 \( I_{org} \)의 MSE는 0.035입니다.
이러한 차이는 비, 안개와 같은 복합 변환 및 커버리지 유도 검색 중에 생성된 다른 합성 이미지에서도 존재합니다.
따라서, 전반적으로 \( I_{err} \)는 결함 있는 동작을 나타낼 가능성이 더 높습니다.

![Figure 6](/posts/paper/DeepTest/figure6.png)
    
To reduce such false positives, we only report bugs for the transformations (e.g., small rotations, rain, fog, etc.) where the correct output should not deviate much from the labels of the corresponding seed images.
Thus, we further use a filtration criteria as defined in Equation 3 to identify such transformations by checking whether the MSE of the synthetic images is close to that of the original images.

잘못된 긍정(false positive)을 줄이기 위해, 우리는 올바른 출력이 해당 시드 이미지의 레이블과 크게 벗어나지 않아야 하는 변환(예: 작은 회전, 비, 안개 등)에 대해서만 버그를 보고합니다.
따라서 우리는 합성 이미지의 MSE가 원본 이미지의 MSE와 가까운지 여부를 확인하여 이러한 변환을 식 3에 정의된 필터링 기준을 사용하여 식별합니다.

\[ | \text{MSE}(\text{trans}, \text{param}) - \text{MSE}_{org} | \leq \epsilon \]

Table 7 shows the number of such erroneous cases by varying two thresholds: ε and λ—a higher value of λ and lower value of ε makes the system report fewer bugs and vice versa.
For example, with a λ of 5 and ε of 0.03, we report 330 violations for simple transformations.

표 7은 두 가지 임계값 \(\epsilon\)과 \(\lambda\)를 변경하여 이러한 오류 사례의 수를 보여줍니다.
\(\lambda\) 값이 높고 \(\epsilon\) 값이 낮을수록 시스템은 더 적은 버그를 보고하며, 그 반대도 마찬가지입니다.
예를 들어, \(\lambda\) = 5와 \(\epsilon\) = 0.03인 경우 간단한 변환에 대해 330개의 위반을 보고합니다.

![Table 7](/posts/paper/DeepTest/table7.png)

Table 8 further elaborates the result for different models for λ = 5 and ε = 0.03, as highlighted in Table 7.
Interestingly, some models are more prone to erroneous behaviors for some transformations than others.
For example, Rambo produces 23 erroneous cases for shear, while the other two models do not show any such cases.
Similarly, DeepTest finds 650 instances of erroneous behavior in Chauffeur for rain but only 64 and 27 for Epoch and Rambo respectively. 

표 8은 표 7에서 강조된 \(\lambda\) = 5 및 \(\epsilon\) = 0.03에 대한 다양한 모델의 결과를 더욱 상세히 설명합니다.
흥미롭게도, 일부 모델은 특정 변환에 대해 다른 모델보다 오류 동작을 더 많이 발생시킵니다.
예를 들어, Rambo 모델은 기울임(shear)에 대해 23개의 오류 사례를 발생시키는 반면, 다른 두 모델은 이러한 사례를 전혀 보여주지 않습니다.
마찬가지로, DeepTest는 비(rain)에 대해 Chauffeur 모델에서 650개의 오류 동작을 찾았지만 Epoch와 Rambo에서는 각각 64개와 27개만 찾았습니다.

![Table 8](/posts/paper/DeepTest/table8.png)
    
We also manually checked the bugs reported in Table 8 and report the false positives in Figure 8.
It also shows two synthetic images (the corresponding original images) where DeepTest incorrectly reports erroneous behaviors while the model’s output is indeed safe.

우리는 표 8에 보고된 버그를 수동으로 확인하고, 그림 8에 잘못된 긍정(false positive)을 보고합니다.
그림 8은 DeepTest가 잘못된 오류 동작을 보고한 두 개의 합성 이미지(해당 원본 이미지)를 보여주며, 실제로 모델의 출력은 안전합니다.

![Figure 8](/posts/paper/DeepTest/figure8.png)

**Result 4**: 뉴런 커버리지를 기반으로 한 합성 이미지를 통해, DeepTest는 세 모델이 예측한 대로 1,000개 이상의 오류 동작을 성공적으로 탐지했으며, 잘못된 긍정 비율이 낮았습니다.

### RQ5. Can retraining DNNs with synthetic images improve accuracy?
Here we check whether retraining the DNNs with some of the synthetic images generated by DeepTest helps in making the DNNs more robust.
We used the images from HMB_3.bag [16] and created their synthetic versions by adding the rain and fog effects.
We retrained the Epoch model with randomly sampled 66% of these synthetic inputs along with the original training data.
We evaluated both the original and the retrained model on the rest 34% of the synthetic images and their original versions.
In all cases, the accuracy of the retrained model improved significantly over the original model as shown in Table 9.

우리는 DeepTest가 생성한 일부 합성 이미지를 사용하여 DNN을 재훈련하는 것이 DNN을 더 견고하게 만드는 데 도움이 되는지 확인합니다.
HMB_3.bag [16]의 이미지를 사용하여 비와 안개 효과를 추가하여 합성 버전을 만들었습니다.
우리는 Epoch 모델을 원본 훈련 데이터와 함께 이러한 합성 입력의 66%를 무작위로 샘플링하여 재훈련했습니다.
나머지 34%의 합성 이미지와 그 원본 버전에 대해 원본 모델과 재훈련된 모델을 평가했습니다.
모든 경우에서 재훈련된 모델의 정확도가 원본 모델에 비해 크게 향상되었습니다(표 9 참조).

![Table 9](/posts/paper/DeepTest/table9.png)

**Result 5**: DeepTest가 생성한 합성 데이터를 사용하여 DNN을 재훈련함으로써 DNN의 정확도를 최대 46%까지 향상시킬 수 있습니다.

## 6. THREATS TO VALIDITY
DeepTest generates realistic synthetic images by applying different image transformations on the seed images.
However, these transformations are not designed to be exhaustive and therefore may not cover all realistic cases.
While our transformations like rain and fog effects are designed to be realistic, the generated pictures may not be exactly reproducible in reality due to a large number of unpredictable factors, e.g., the position of the sun, the angle and size of the rain drops. etc.
However, as the image processing techniques become more sophisticated, the generated pictures will get closer to reality.
A complete DNN model for driving an autonomous vehicle must also handle braking and acceleration besides the steering angle.
We restricted ourselves to only test the accuracy of the steering angle as our tested models do not support braking and acceleration yet.
However, our techniques should be readily applicable to testing those outputs too assuming that the models support them.

DeepTest는 시드 이미지에 다양한 이미지 변환을 적용하여 현실적인 합성 이미지를 생성합니다.
그러나 이러한 변환은 모든 현실적인 경우를 포괄하도록 설계되지 않았기 때문에 완전하지 않을 수 있습니다.

비와 안개 효과와 같은 우리의 변환은 현실적으로 설계되었지만, 생성된 이미지가 현실에서 정확하게 재현되지는 않을 수 있습니다.
이는 태양의 위치, 빗방울의 각도와 크기 등 예측할 수 없는 많은 요인 때문입니다.
그러나 이미지 처리 기술이 더욱 정교해짐에 따라 생성된 이미지는 현실에 더 가까워질 것입니다.

자율 주행 차량을 위한 완전한 DNN 모델은 조향 각도 외에도 제동과 가속을 처리해야 합니다.
우리는 테스트한 모델이 아직 제동과 가속을 지원하지 않기 때문에 조향 각도의 정확성만 테스트하는 것으로 제한했습니다.
그러나 모델이 이를 지원한다고 가정하면 우리의 기술은 이러한 출력도 테스트하는 데 쉽게 적용할 수 있을 것입니다.

## 7. RELATED WORK
**머신 러닝 테스트 및 검증**
전통적인 머신 러닝 시스템 평가 방법은 주로 수동으로 라벨링된 데이터셋에서 무작위로 추출한 테스트 입력과 임시 시뮬레이션을 통해 정확도를 측정합니다 [11, 20, 82].
그러나 모델 내부에 대한 지식 없이 이러한 블랙박스 테스트 패러다임은 예상치 못한 동작을 유발할 수 있는 다양한 코너 케이스를 찾지 못합니다 [26, 70].

Pei 등 [70]은 여러 DNN 간의 불일치를 유발할 수 있는 입력을 체계적으로 찾기 위해 DeepXplore라는 화이트박스 차분 테스트 알고리즘을 제안했습니다.
그들은 DNN의 내부 논리가 얼마나 테스트되었는지를 측정하기 위한 체계적인 지표로 뉴런 커버리지를 도입했습니다.
반면에, 우리의 그레이박스 방법은 단일 DNN에서 뉴런 커버리지를 이용한 유도 테스트 생성에 사용하고, 변형 관계를 활용하여 여러 DNN을 필요로 하지 않고 오류 동작을 식별합니다.

또 다른 최근 연구는 다양한 안전 속성에 대해 DNN을 검증하는 가능성을 탐구했습니다 [48, 51, 71].
그러나 이러한 기술 중 어느 것도 실제 크기의 DNN에 대해 풍부한 속성 집합을 검증할 수는 없습니다.
반면에, 우리의 기술은 최첨단 DNN을 체계적으로 테스트하여 안전에 중요한 오류 동작을 확인할 수 있지만, 이론적인 보장을 제공하지는 않습니다.

**적대적 머신 러닝**
많은 프로젝트가 테스트 시 머신 러닝 모델을 공격하여 예상치 못한 오류를 유발하는 데 성공했습니다.
보다 구체적으로, 이러한 공격은 원본 버전에서 최소한으로 변경되었을 때 머신 러닝 분류기에 의해 다르게 분류되는 입력을 찾는 데 중점을 둡니다.
이러한 유형의 공격은 이미지 인식 [37, 40, 52, 55, 62, 63, 65, 66, 78], 얼굴 감지/검증 [75, 81], 악성 코드 탐지 [28, 42, 54, 85], 텍스트 분석 [59, 67] 등 다양한 작업에 영향을 미치는 것으로 알려져 있습니다.
여러 이전 연구는 이러한 공격에 대한 방어를 다양한 효과로 탐구했습니다 [29, 32, 35, 38, 41, 43, 48, 57, 64, 68, 74, 77, 80, 84, 86].

요약하자면, 이 연구 분야는 최소한의 노이즈를 추가하여 잘못된 예측을 유도하는 특정 유형의 오류 동작을 찾으려고 합니다.
반면, 우리는 뉴런 커버리지를 최대화하여 주어진 DNN을 체계적으로 테스트하고 다양한 코너 케이스 동작을 찾습니다.
또한, 우리는 실제로 발생할 수 있는 현실적인 조건을 찾는 데 중점을 둡니다.

**테스트 증폭**
전통적인 소프트웨어를 위한 테스트 케이스 생성 및 증폭 기법에 대한 많은 연구가 있으며, 이들은 일부 시드 입력에서 테스트 케이스를 자동으로 생성하고 코드 커버리지를 증가시킵니다.
개별 연구를 요약하는 대신, 관심 있는 독자는 Anand et al. [27], McMinn et al. [56], Pasareanu et al. [69]의 설문 조사를 참조하시기 바랍니다.
이러한 접근 방식과 달리, DeepTest는 DNN에서 작동하도록 설계되었습니다.

**변형 테스트**
변형 테스트 [33, 87]는 수동 사양이 없는 상황에서 테스트 오라클을 생성하는 방법입니다.
변형 테스트는 서로 다른 입력으로 테스트 프로그램을 여러 번 실행한 결과에서 도메인 특정 변형 관계를 위반하는지 감지하여 버그 동작을 식별합니다.
예를 들어, 두 입력 \(a\)와 \(b\)를 더하는 프로그램 \(p\)에 대한 변형 속성의 예는 \(p(a, b) = p(a, 0) + p(b, 0)\)일 수 있습니다.
변형 테스트는 과거에 지도 학습 및 비지도 학습 머신 러닝 분류기를 테스트하는 데 사용되었습니다 [60, 83].
반면, 우리는 자율 주행 자동차의 도메인에서 새로운 변형 관계를 정의합니다.
이는 이전에 테스트된 분류기와 달리 연속적인 조향 각도를 생성하므로, 회귀 작업입니다.

## 8. CONCLUSION
In this paper, we proposed and evaluated DeepTest, a tool for automated testing of DNN-driven autonomous cars.
DeepTest maximizes the neuron coverage of a DNN using synthetic test images generated by applying different realistic transformations on a set of seed images.
We use domain-specific metamorphic relations to find erroneous behaviors of the DNN without detailed specification.
DeepTest can be easily adapted to test other DNN-based systems by customizing the transformations and metamorphic relations.
We believe DeepTest is an important first step towards building robust DNN-based systems.

이 논문에서 우리는 DNN 기반 자율 주행 자동차를 자동으로 테스트하는 도구인 DeepTest를 제안하고 평가했습니다.
DeepTest는 시드 이미지 세트에 다양한 현실적인 변환을 적용하여 생성된 합성 테스트 이미지를 사용하여 DNN의 뉴런 커버리지를 최대화합니다.
우리는 도메인 특정 변형 관계를 사용하여 상세한 사양 없이 DNN의 오류 동작을 발견합니다.
DeepTest는 변환과 변형 관계를 사용자 정의하여 다른 DNN 기반 시스템을 테스트하는 데 쉽게 적용할 수 있습니다.
우리는 DeepTest가 견고한 DNN 기반 시스템을 구축하기 위한 중요한 첫걸음이라고 믿습니다.
