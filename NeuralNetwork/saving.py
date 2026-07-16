import pickle

def save_autoencoder(path, encoder, mu, log_sigma, decoder):
    model_state = {
        'encoder': encoder,
        'mu': mu,
        'log_sigma': log_sigma,
        'decoder': decoder
    }

    with open(path, 'wb') as f:
        pickle.dump(model_state, f)
    print("Saved model")

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)