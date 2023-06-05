from unittest.case import _AssertRaisesContext
from openai_api_server import get_gen_params, update_params


def test_get_gen_params_gpt_4_0314():
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
    gen_params = get_gen_params("gpt-4-0314", messages=messages, temperature=0.9, max_tokens=10, top_p=1.0, echo=False, stream=False, stop=["\n"])
    assert gen_params["model"] == "gpt-4-0314"

def test_get_gen_params_gpt4():
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
    with _AssertRaisesContext(ValueError):
        get_gen_params("gpt-4", messages=messages, temperature=0.9, max_tokens=10, top_p=1.0, echo=False, stream=False, stop=["\n"])

def test_update_params():
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
    gen_params = get_gen_params("gpt-4-0314", messages=messages, temperature=0.9, max_tokens=10, top_p=1.0, echo=False, stream=False, stop=["\n"])
    update_params(gen_params, updated_model_name="gpt-3.5-turbo-0301")
    assert gen_params["model"] == "gpt-3.5-turbo-0301"


test_get_gen_params_gpt_4_0314()
test_get_gen_params_gpt4()