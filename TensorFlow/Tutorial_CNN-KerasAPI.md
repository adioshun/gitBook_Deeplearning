## model - comple - fit -evaluate process

```python 

import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras




def main():
    num_classes = 10
    batch_size = 32
    epochs = 1

    # build model and optimizer
    model = ResNet([2, 2, 2], num_classes)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.build(input_shape=(None, 28, 28, 1))
    print("Number of variables in the model :", len(model.variables))
    model.summary()

    # train
    model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test_ohe), verbose=1)

    # evaluate on test set
    scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)




if __name__ == '__main__':
    main()

```






