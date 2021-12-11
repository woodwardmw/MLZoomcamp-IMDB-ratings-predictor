import datetime
import tensorflow as tf


def run_experiment(model, tf_train_dataset, tf_val_dataset, params):
    
    NUM_STEPS = params['max_rows'] // params['batch_size']
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_callback =     tf.keras.callbacks.ModelCheckpoint(
            "distilBERT_{epoch:02d}_{val_mse:.3f}.h5",
            monitor="val_loss",
            save_best_only=True,
            mode='min'
        )

    # Train the model
    train_history1 = model.fit(
        tf_train_dataset,    
        epochs = params['epochs'],
        batch_size = params['batch_size'],
        steps_per_epoch = NUM_STEPS,
        validation_data = tf_val_dataset,
        verbose=2,
        callbacks=[tensorboard_callback]
    )