from omegaconf import DictConfig
from pathlib import Path
from calf import device, logger, parse_config
from calf.utils.corpus import UDTreebankCorpus
from calf.utils.dataset import Dataset, build_dataloader
from calf.modules.huggingface import HuggingfaceTokenizer, HuggingfaceModel, ModelType
from .ud_transform import UDTreebankTransform


def init(cfg: DictConfig) -> None:
    cfg = parse_config(cfg=cfg,
                       config_path=str(Path(__file__).parent),
                       config_file="test_pythia.yaml")
    for model_params in cfg.pythia_models:
        tokenizer = HuggingfaceTokenizer(model_name=model_params.name,
                                         revision=model_params.get("revision", None))
        logger.info(f"Tokenizer: {tokenizer.name} "
                    f"{model_params.get('revision', None)} {len(tokenizer)}")
        model = HuggingfaceModel(model_name=model_params.name,
                                 model_type=ModelType.CausalLM,
                                 revision=model_params.get("revision", None),
                                 n_layers=1)
        logger.info(f"Model: {model.name} {model_params.get('revision', None)}")


def transform(cfg: DictConfig) -> None:
    cfg = parse_config(cfg=cfg,
                       config_path=str(Path(__file__).parent),
                       config_file="test_pythia.yaml")
    for model_params in cfg.pythia_models:
        tokenizer = HuggingfaceTokenizer(model_name=model_params.name,
                                         revision=model_params.get("revision", None))
        transform = UDTreebankTransform(tokenizer=tokenizer)
        logger.info(transform.name)
        corpus = UDTreebankCorpus()
        dataset = transform.load(corpus)
        for i, data in enumerate(transform.transform(dataset)):
            if i == 0:
                logger.info(data)
                break


def dataset(cfg: DictConfig) -> None:
    cfg = parse_config(cfg=cfg,
                       config_path=str(Path(__file__).parent),
                       config_file="test_pythia.yaml")
    for model_params in cfg.pythia_models:
        tokenizer = HuggingfaceTokenizer(model_name=model_params.name,
                                         revision=model_params.get("revision", None))
        transform = UDTreebankTransform(tokenizer=tokenizer)
        for corpus_params in cfg.ud_corpora:
            corpus = UDTreebankCorpus(lang=corpus_params.lang,
                                      genre=corpus_params.genre,
                                      split=corpus_params.split)
            dataset = Dataset(corpus=corpus, transform=transform, cache=True)
            logger.info(f"lang: {corpus_params.lang}, "
                        f"genre: {corpus_params.genre}, "
                        f"split: {corpus_params.split}, "
                        f"length: {len(dataset)}")


def tokenize(cfg: DictConfig) -> None:
    cfg = parse_config(cfg=cfg,
                       config_path=str(Path(__file__).parent),
                       config_file="test_pythia.yaml")
    for model_params in cfg.pythia_models:
        tokenizer = HuggingfaceTokenizer(model_name=model_params.name,
                                         revision=model_params.get("revision", None))
        if tokenizer.pad_token is None:
            special_tokens = {
                "pad_token": tokenizer.eos_token
            }
            tokenizer.add_special_tokens(special_tokens)
        logger.info(tokenizer.pad_token_id)
        logger.info(tokenizer.pad_token)
        text = ['Danksdf', 'Morgan', 'told', 'himself', 'he', 'would', 'forget', 'Ann',
                'Turner', '.']
        data = tokenizer(text)
        logger.info(data["input_ids"])
        logger.info(data["segment_spans"])


def embed(cfg: DictConfig) -> None:
    cfg = parse_config(cfg=cfg,
                       config_path=str(Path(__file__).parent),
                       config_file="test_pythia.yaml")
    for model_params in cfg.pythia_models:
        tokenizer = HuggingfaceTokenizer(model_name=model_params.name,
                                         revision=model_params.get("revision", None))
        model = HuggingfaceModel(model_name=model_params.name,
                                 model_type=ModelType.CausalLM,
                                 output_hidden_states=True,
                                 revision=model_params.get("revision", None),
                                 n_layers=1)
        model.eval()
        model.to(device)
        transform = UDTreebankTransform(tokenizer=tokenizer)
        corpus = UDTreebankCorpus()
        dataloader, dataset, sample = build_dataloader(corpus=corpus,
                                                       transform=transform,
                                                       cache=True,
                                                       batch_size=cfg.batch_size)
        for i, batch in enumerate(dataloader):
            if i == 0:
                embedding = model(batch["input_ids"], batch["attention_mask"])
                logger.info(embedding)
                break
