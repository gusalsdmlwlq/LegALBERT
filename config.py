import argparse

class Config:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="data/bert", type=str)
    parser.add_argument("--vocab_path", default="assets/kor-vocab.txt", type=str)
    parser.add_argument("--config_path", default="assets/albert_config.txt", type=str)

    # training
    parser.add_argument("--masking_rate", default=0.15, type=float)
    parser.add_argument("--masking_unk_rate", default=0.8, type=float)
    parser.add_argument("--masking_random_rate", default=0.1, type=float)
    parser.add_argument("--mask_token_id", default=4, type=int)

    # ALBERT config
    parser.add_argument("--pad_token_id", default=0, type=int)
    parser.add_argument("--bos_token_id", default=2, type=int)
    parser.add_argument("--eos_token_id", default=3, type=int)
    parser.add_argument("--hidden_dropout_prob", default=0, type=int)
    parser.add_argument("--attention_probs_dropout_prob", default=0, type=int)
    parser.add_argument("--classifier_dropout_prob", default=0.1, type=int)
    parser.add_argument("--embedding_size", default=128, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--hidden_act", default="gelu_new", type=str)
    parser.add_argument("--initializer_range", default=0.02, type=int)
    parser.add_argument("--inner_group_num", default=1, type=int)
    parser.add_argument("--intermediate_size", default=2048, type=int)
    parser.add_argument("--layer_norm_eps", default=1e-12, type=int)
    parser.add_argument("--max_position_embeddings", default=512, type=int)
    parser.add_argument("--model_type", default="albert", type=str)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--num_hidden_groups", default=1, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--type_vocab_size", default=2, type=int)
    parser.add_argument("--vocab_size", default=30100, type=int)