#Opensource libraries:
 - Opencv3.4
 - Tensorflow 1.4 com suporte a GPUs
 - Python 2
 
 PS: a maioria dos arquivos da parte de Resnet foi adaptada de outro código, por isso os nomes dos arquivos e de muitas variáveis estão com "cifar10". Como a troca tornaria necessário debug até que todos os lugares se acertassem e o prazo era curto, preferiu-se por deixar assim.
 
PASTA
correct_gt: códigos para correção do Ground Truth fornecido
correct_gt/correct_gt.py: corrige o GT de arquivos com nome fornecido
correct_gt/visualize_quality_of_corrected.py: visualiza GT corrigido

PASTA
FullSkinDataset: dataset completo, com GT original e corrigido

PASTA
Generate_Jaccard: gera o Jaccard index para imagens de prediction e GT fornecidas

PASTA
Resnet_Code: Requisito 1 para Resnet

PASTA
Resnet_Code/Generate_TFRecords: gera arquivos .tfrecords usados no treinamento e avaliação
Resnet_Code/Generate_TFRecords/convert_full_test_to_tfrecords.py: gera .tfrecords de teste
Resnet_Code/Generate_TFRecords/convert_train_to_tfrecords_bigger_batches.py: gera .tfrecords de treino

PASTA
Resnet_Code/Resnet_on_sfa: Utiliza os .tfrecors e a rede para treinamento e validação
Resnet_Code/Resnet_on_sfa/cifar10.py: Arquivo com o modelo e funções relativas à rede, não deve ser chamado diretamente
Resnet_Code/Resnet_on_sfa/cifar10_multi_gpu_train.py: Arquivo que processa os .tfrecords e chama cifar10.py para realizar o treinamento. É responsável também pelo gerenciamento multi GPU
Resnet_Code/Resnet_on_sfa/cifar10_input.py: É usado apenas para definição de FLAGS chamadas por outros arquivos. O código aí presente é resíduo do original e pode ser descartado
Resnet_Code/Resnet_on_sfa/cifar10_eval_on_training_set.py: Faz avaliação dos arquivos de teste com a rede já treinada

PASTA
Resnet_on_full_dataset: Requisito 2 para Resnet

PASTA
Resnet_on_full_dataset/Generate_TFRecords: gera arquivos .tfrecords usados no treinamento e avaliação
Resnet_on_full_dataset/Generate_TFRecords/convert_full_test_to_tfrecords_bigger_batch.py: gera .tfrecords de teste usando todos os pixels de cada imagem de teste
Resnet_on_full_dataset/Generate_TFRecords/convert_test_to_tfrecords_bigger_batch.py: gera .tfrecords de teste usando subsets dos pixels
Resnet_on_full_dataset/Generate_TFRecords/convert_train_to_tfrecords_bigger_batch.py: gera .tfrecords de treinamento usando subsets dos pixels
Resnet_on_full_dataset/Generate_TFRecords/convert_validation_to_tfrecords_bigger_batch.py: gera .tfrecords de validação usando subsets dos pixels

PASTA
Resnet_on_full_dataset/Resnet_on_full_sfa: Arquivos operam de maneira semelhante à pasta do requisito 1, com exceção do fato de que agora há dois arquivos de validação, um que desenha as imagens e outro não


PASTA
SkinDataset: dataset reduzido para requisito 1, com GT original e corrigido
