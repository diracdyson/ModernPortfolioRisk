

import tf
from tf import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
import time 
import keras_tuner

class DeepNN(keras_tuner.HyperModel):
    def build( self,hp) -> keras_tuner.HyperModel:
        
        model=keras.Sequential(
      [tf.keras.layers.Dense(units=hp.Int(f"units_1", min_value=2, max_value=8, step=2),
                                          activation=hp.Choice(f"activation_1", ["relu", "tanh"])),
       #tf.keras.layers.BatchNormalization(),
       #tf.keras.layers.Dropout(rate=hp.Float("dropout1", 0.0,.02,.002)),
        tf.keras.layers.Dense(units=hp.Int(f"units_2", min_value=2, max_value=8,step=2),activation=hp.Choice(f"activation_2", ["relu", "tanh"])),
       #tf.keras.layers.BatchNormalization(),
       
       #tf.keras.layers.BatchNormalization(),
       #f.keras.layers.Dropout(rate=hp.Float("dropout7", .01,.02,.002)),
        tf.keras.layers.Dense(1,activation='sigmoid')])


        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss="mse",metrics =["accuracy"])

        return model
    

    def fit(self,hp,model,x,y,xval,yval, batch_size,epochs,**kwargs):
        #x,xval,y,yval=train_test_split(training.drop('churn',axis=1).values,training.churn.values.reshape(-1,1),test_size=hp.Float(f"val_split", 0.1,0.4,.02))
        epochs=hp.Int(f"epochs",100,500,100)
        batch_size=hp.Int(f"batch_size",10,30,5)
        #loss_fn=keras.losses.MeanSquaredError()
        loss_fn=tf.keras.losses.MSE(from_logits=False)
        lr=hp.Float(f"learning_rate", .0001,.03,.0001)
        optimizer=keras.optimizers.Adam(learning_rate=lr)
        #train_acc_metric=keras.metrics.MeanSquaredError()
        train_acc_metric = tf.keras.metrics.MSE(from_logits=False)
        #val_acc_metric=keras.metrics.MeanSquaredError()
        val_acc_metric=tf.keras.metrics.MSE(from_logits=False)
        train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((xval, yval))
        val_dataset = val_dataset.batch(batch_size)
        patience=5
        delta=0.001
        loss_history=deque(maxlen=patience+1)

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss_value = loss_fn(y, logits)
        # Add any extra losses created during the forward pass.
                loss_value += sum(model.losses)
            
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_acc_metric.update_state(y, logits)
            return loss_value
        #@tf.function
        #def train_step(x,y):
        #    with tf.GradientTape() as tape:
       #         logits=model(x,training=True)
        #        loss_value=loss_fn(y,logits)
        #    grads=tape.gradient(loss_value,model.trainable_weights )
        #    train_acc_metric.update_state(y,logits)
        #    return loss_value
        
        @tf.function
        def test_step(x, y):
            val_logits = model(x, training=False)
            val_acc_metric.update_state(y, val_logits)
        
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

    # Iterate over the batches of the dataset.
            for step, (x_batch_train,y_batch_train ) in enumerate(train_dataset):
                loss_value = train_step(x_batch_train, y_batch_train)

        # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                         )
                    print("Seen so far: %d samples" % ((step + 1) * batch_size))

    # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()


            for x_batch_val, y_batch_val in val_dataset:
                test_step(x_batch_val, y_batch_val)
            loss_history.append(train_acc_metric.result())
            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))
            
            if len(loss_history) > patience:
                if loss_history.popleft()*delta < min(loss_history):
                    print(f'\nEarly stopping. No improvement of more than {delta:.5%} in '
                  f'validation loss in the last {patience} epochs.')
                    break 
        return {
            "metric_a": loss_value,
            "metric_b": val_acc,
        }



    def predict(self,hp,model,x,*args,**kwargs) ->np.array():
        return model.predict(x,*args,**kwargs)
