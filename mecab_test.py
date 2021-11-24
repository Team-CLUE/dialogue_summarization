#import mecab
from konlpy.tag import Mecab

if __name__ == '__main__':
    mecab = Mecab()
    print(mecab.pos('이순신은 조선의 무신이다.'))