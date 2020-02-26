import config as cfg
from model import Model


def main():
    model = Model(cfg.encoder_layers, cfg.decoder_layers, cfg.hidden_dim, cfg.batch_size, cfg.learning_rate, cfg.dropout, init_train = True)
    print("simple test")
    model.train_epoch(1)


if __name__== '__main__':
    main()