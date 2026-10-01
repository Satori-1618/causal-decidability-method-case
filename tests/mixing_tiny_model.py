"""A tiny, randomly initialised Gemma 2 and a word-level tokenizer, built locally.

Test fixture for the Round 1 runner (Gur-Arieh et al. application). Nothing is
downloaded and no Gemma file is read: the tokenizer mimics Gemma's conventions (a
'<bos>' token added by default, '▁' marking a leading space, a Gemma-style chat
template) over a small vocabulary built from the prompts it must encode.
"""
import itertools

TINY_SPEC = {
    "constant": "TINY_MUSIC",
    "name": "tiny_music",
    "categories": ["Musician", "Genre", "Instrument"],
    "items": {
        "Musician": ["John", "Mary", "Bob", "Sam", "Alex", "Emma", "David", "Sarah", "Tom"],
        "Genre": ["jazz", "rock", "blues", "folk", "pop", "funk", "soul", "punk", "disco"],
        "Instrument": ["piano", "guitar", "violin", "drums", "flute", "trumpet", "bass", "organ", "horn"],
    },
    "definitions": {"row_default": "{Musician} performed {Genre} music on the {Instrument}"},
    "prefix": "At the music festival, ",
    "capitalize_first_clause": False,
    "queries": {"Q:Instrument_Musician A:Genre": {
        "question": "Respond in one word, only the answer and nothing else: "
                    "What music did {Musician} play on the {Instrument}?",
        "answer_category": "Genre"}},
    "max_new_tokens": 3,
}
MULTI_TOKEN_GENRE = "hip hop"  # used to show that pools drop multi-token entities
CHAT_TEMPLATE = ("{{ bos_token }}{% for message in messages %}<start_of_turn>{{ message['role'] }}\n"
                 "{{ message['content'] }}<end_of_turn>\n{% endfor %}"
                 "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")
SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]


def tiny_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    pre = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"\n"), behavior="isolated"),
        pre_tokenizers.Split(Regex(r"[,.?:]"), behavior="isolated"),
        pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never"),
    ])
    items = TINY_SPEC["items"]
    first = {category: values[0] for category, values in items.items()}
    corpus = [TINY_SPEC["prefix"], TINY_SPEC["definitions"]["row_default"].format(**first),
              TINY_SPEC["queries"]["Q:Instrument_Musician A:Genre"]["question"].format(**first),
              ". " + TINY_SPEC["queries"]["Q:Instrument_Musician A:Genre"]["question"].format(**first),
              " Answer:", "user\n", "model\n", " and", ", and", " " + MULTI_TOKEN_GENRE]
    corpus += [prefix + e for values in items.values() for e in values for prefix in ("", " ")]
    corpus += [e.capitalize() for e in items["Genre"]]
    words = sorted({piece for text in corpus for piece, _ in pre.pre_tokenize_str(text)})
    vocab = {token: i for i, token in enumerate(SPECIALS + [w for w in words if w not in SPECIALS])}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre
    core.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="never")
    core.add_special_tokens(SPECIALS)
    core.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", vocab["<bos>"])])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<bos>", eos_token="<eos>",
                                        unk_token="<unk>", pad_token="<pad>",
                                        additional_special_tokens=["<start_of_turn>", "<end_of_turn>"])
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def tiny_model(vocab_size, seed=0, layers=4, dtype=None):
    import torch
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(seed)
    config = Gemma2Config(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=512,
        sliding_window=256, query_pre_attn_scalar=16, pad_token_id=0, eos_token_id=1, bos_token_id=2,
        attn_implementation="eager", final_logit_softcapping=30.0, attn_logit_softcapping=50.0)
    model = Gemma2ForCausalLM(config)
    # Random init gives near-zero logits; scale the embedding so outputs differ by input.
    with torch.no_grad():
        model.model.embed_tokens.weight.mul_(20.0)
    if dtype is not None:
        model = model.to(dtype)
    return model.eval()


def seeds(start, count):
    return list(itertools.islice(itertools.count(start), count))
