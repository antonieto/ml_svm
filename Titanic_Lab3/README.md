# Lab 3: Titanic

------------------------------------------------------------------------

> En esta práctica vamos a encontrar el mejor clasificador para los datos del Titanic publicados en Kaggle (<https://www.kaggle.com/c/titanic>), que también podemos encontrarlos en el Campus Virtual. La forma en la que vamos a abordar el problema es la siguiente: En primer lugar, realizaremos un examen del dataset y estudiaremos los atributos y si aparecen datos faltantes. Como final de la fase de preprocesamiento obtendremos un dataset en el que se hayan eliminado los atributos innecesarios y se haya solucionado los atributos faltantes. En segundo lugar, entrenaremos a  cada uno de los clasificadores (Rpart, nnet, e1071) usando validación cruzada. Obtendremos el accuracy y el área bajo la curva para cada clasificador. En tercer lugar, vamos a entrenar a cada uno de los clasificadores, usando validación cruzada y modificando sus parámetros como CP , size o el tipo de kernel, con el objetivo de encontrar el clasificador mejor de su clase. Por último, obtendremos el mejor clasificador de todos y realizaremos una predicción con el mismo usando los datos de test de Titanic.

## Dataset analysis

- Attributes
- Label
- More..

### Data preprocessing

- Which attributes are useful?
- Which are not?
- Are there missing attributes?

#### Adding a `title` feature.
By analyzing the contents in the *name* feature, we noticed the names contain titles. We can extract these titles by processing the names and trying to match the name to one of several possible titles.

#### Relationship between Age and Class
$$
  I_{Age.Class} = I_{Age} * I_{Class}
$$
The intuition for

--- 

## Training / Model Fitting

Describes which models we are going to fit to our problem.

### Decision tree: `rpart`

- Code for training a decision tree using `rpart`


### Neural Network (a.k.a. *Perceptron*): `nnet`

- Code for training
- Library used
- Network architecture

### Support Vector Macine: `e1071`

- Training code
- `kernlab` vs `e1071`?

---

## Fine tuning and comparisons

### Complexity parameter (decision tree)

- Optimizing for a better CP

### Kernel (SVM)

- Fine tunning for a better kernel function

### ROC Curves

- Derived ROC curves

### Cross Validation

- Compare models and accuracy


