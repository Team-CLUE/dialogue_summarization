
from transformers import AutoTokenizer

def get_tokenizer(model_name:str) -> AutoTokenizer:
    '''
        Arguments:
            model_name: str 
                허깅페이스에 있는 pretrained toknizer의 모델 이름

        Return
            AutoTokenizer

        Summary:
            허깅페이스에서 Tokenizer를 로딩하고, 정의한 special token을 추가하여 반환
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # special_tokens_dict = {'additional_special_tokens': ['#@URL#','#@이름#','#@계정#','#@신원#','#@전번#',
    #             '#@금융#','#@번호#','#@주소#','#@소속#','#@기타#', '#@이모티콘#', '#@시스템#사진', '#@시스템#검색',  '#@시스템#지도#', '#@시스템#기타#', '#@시스템#파일#',
    #             '#@시스템#동영상#', '#@시스템#송금#', '#@시스템#삭제#']}
    # tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer