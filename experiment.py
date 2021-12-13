import datetime
import tensorflow as tf


def run_experiment(model, tf_train_dataset, tf_val_dataset, params):
    
    NUM_STEPS = params['max_rows'] // params['batch_size']
    checkpoint_callback =     tf.keras.callbacks.ModelCheckpoint(
            "ratings_predictor_trained_epoch_{epoch:02d}_mse_{val_mse:.3f}.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode='min'
        )
    # Train the model
    train_history = model.fit(
        tf_train_dataset,    
        epochs = params['epochs'],
        batch_size = params['batch_size'],
        steps_per_epoch = NUM_STEPS,
        validation_data = tf_val_dataset,
        verbose=2,
        callbacks=[checkpoint_callback]
    )
    return train_history, model