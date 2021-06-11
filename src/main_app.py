import spacy
from spacy_langdetect import LanguageDetector
import streamlit as st
import pandas as pd
import io
import simplejson
import base64
import dl_translate as dlt
import nltk
import newspaper
from cleantext import clean

model_path = 'en_core_web_sm'
nlp = spacy.load(model_path)
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)


m2m_100_mapping_code_path = "input_file/m2m100.json"
mbart_50_mapping_code_path = "input_file/mbart50.json"
languages_support_path = 'input_file/languages_support.json'

logo_image_path = "input_file/lang_translate_image.jpeg".splitlines()

# setting the Page Layout
st.set_page_config(page_title='Translator APP', page_icon='☮️', layout='centered',initial_sidebar_state='expanded')




@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def mbart_50_download_model():
    """
    It's take time in loading browser for the first time, model size is big
    reading the model object
    :return: mbart50 nodel object
    """
    return dlt.TranslationModel("mbart50")
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def m2m_100_download_model():
    """
    It's take time in loading browser for the first time, model size is big
    reading the model object
    :return: m2m_100 nodel object
    """
    return dlt.TranslationModel()



def load_config_file(filename):
    """
    :param json file path to read:
    :return: json object
    """
    if filename is not None:
        try:
            with io.open(filename, encoding='utf-8') as f:
                file_config = simplejson.loads(f.read())
                return file_config
        except ValueError as e:
            print("Failed to read configuration file '{}'. Error: {}".format(filename, e))

m2m_100_json = load_config_file(m2m_100_mapping_code_path)
mbart_50_json = load_config_file(mbart_50_mapping_code_path)
languages_support_json = load_config_file(languages_support_path)

def query_fetch(language_code,ref_Num):
    print(languages_support_json)
    print('lang code: '+ str(language_code),'refNum: '+ ref_Num)
    result = []
    try:
        for node in languages_support_json:
            if node['language'] == 'fr' and node['ref_num'] == 'm2m_100':
                    print(str(node))
                    result.append(node)
    except:
        print('Unable to fetch data from query_fetch')
    return result

def extraxt_text(translation):
   return ' '.join([node['translation'] for node in translation['translations']]).strip()


def set_lang_code(source):
   if source == 'zh-cn':
      source = 'zh'
   elif source == 'zh-tw':
      source = 'zh-TW'
   return source



def _clean_text_(text):
    """
    For cleaning the text before translation
    :param text:
    :return:
    """
    _text = clean(text=text,
           fix_unicode = True,
           to_ascii = True,
           lower = False,  # lowercase text
           no_line_breaks = False,  # fully strip line breaks as opposed to only normalizing them
           no_urls = True,  # replace all URLs with a special token
           no_emails = True,  # replace all email addresses with a special token
           no_phone_numbers = False,  # replace all phone numbers with a special token
           no_numbers = False,  # replace all numbers with a special token
           no_digits = False,  # replace all digits with a special token
           no_currency_symbols = False,  # replace all currency symbols with a special token
           no_punct = True,  # fully remove punctuation
           replace_with_url = " ",
           replace_with_email = " ",
           lang = "en")
    return _text





def lang_detect_nlp(text,nlp):
   doc = nlp(text)
   # document level language detection. Think of it like average language of the document!
   doc_lang = doc._.language
   # sentence level language detection
   # sent_list=[]
   # for sent in doc.sents:
   #    print(sent, sent._.language)
   #    sent_list.append((sent, sent._.language))
   return doc_lang
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="translated.csv">Download csv file</a>'
def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


# Streamlit code start from here
mbart_50_model_object = mbart_50_download_model()
m2m_100_mode_object = m2m_100_download_model()
st.sidebar.image(logo_image_path,width=100)
st.sidebar.header('Language Translation App')
process_box = st.sidebar.radio('Process Type',['Text','URL'])
if process_box == 'URL':
    st.header('Translate URL')
if process_box == 'Text':
    st.header('Translate Text')


#doc_box = st.sidebar.button('Document')
model_type = st.sidebar.multiselect(label='Select Translation Model',options=['M2M_100','MBart_50'])
for model_node in model_type:
    if process_box == 'Text':
        if model_node == 'MBart_50':
            st.subheader('Model: ' + model_node.replace('_',''))
            source_lang_text_MBart_50, target_lang_text_MBart_50 = st.beta_columns(2)
            source_MBart_50 = source_lang_text_MBart_50.multiselect(label='source language', options=list(mbart_50_json.keys()),
                                                          default='English')
            target_MBart_50 = target_lang_text_MBart_50.multiselect(label='target language', options=list(mbart_50_json.keys()),
                                                          default='Hindi')
            right_input_box_MBart, left_output_box_MBart = st.beta_columns(2)
            # demo_text = "Hackeri?a pe jumatate rom√¢nca ce face furori √Æn SUA la 12 ani: ‚ÄùSistemul de vot rom√¢nesc este mult mai bun dec√¢t cel american‚Äù"
            MBart_demo_text = "I am unmarried and even though have a son who is unworried about this."
            MBart_demo_text = right_input_box_MBart.text_area('Input the text', MBart_demo_text, height=400, max_chars=1000,key='MBart_50')
            mbart_sents = nltk.tokenize.sent_tokenize(MBart_demo_text)  # don't use dlt.lang.ENGLISH
            trans_text_mbart_50 = " "
            if st.button('Translate',key='MBart_50'):
                try:
                    trans_text_mbart_50 = " ".join(mbart_50_model_object.translate(mbart_sents, source=source_MBart_50[0], target=target_MBart_50[0],batch_size=10))
                except:
                    trans_text_mbart_50 = "For this source language, we don't have support right now"
            left_output_box_MBart.text_area('Translation Text', trans_text_mbart_50, height=400, max_chars=1000,key='MBart_50')

        elif model_node == 'M2M_100':
            st.subheader('Model: ' + model_node.replace('_',''))
            source_lang_text_M2M_100, target_lang_text_M2M_100 = st.beta_columns(2)
            source_M2M_100 = source_lang_text_M2M_100.multiselect(label='source language',
                                                                    options=list(m2m_100_json.keys()),
                                                                    default='English')
            target_M2M_100 = target_lang_text_M2M_100.multiselect(label='target language',
                                                                    options=list(m2m_100_json.keys()),
                                                                    default='Hindi')
            right_input_box_M2M_100, left_output_box_M2M_100 = st.beta_columns(2)
            # demo_text = "Hackeri?a pe jumatate rom√¢nca ce face furori √Æn SUA la 12 ani: ‚ÄùSistemul de vot rom√¢nesc este mult mai bun dec√¢t cel american‚Äù"
            M2M_100_demo_text = "I am unmarried and even though have a son who is unworried about this."
            M2M_100_demo_text = right_input_box_M2M_100.text_area('Input the text', M2M_100_demo_text, height=400,
                                                              max_chars=1000,key='M2M_100')
            M2M_100_sents = nltk.tokenize.sent_tokenize(M2M_100_demo_text)  # don't use dlt.lang.ENGLISH
            trans_text_m2m_100 = " "
            if st.button('Translate', key='M2M_100'):
                try:
                    trans_text_m2m_100 = " ".join(
                        m2m_100_mode_object.translate(M2M_100_sents, source=source_M2M_100[0],
                                                        target=target_M2M_100[0], batch_size=10))
                except:
                    trans_text_m2m_100 = "For this source language, we don't have support right now"
            left_output_box_M2M_100.text_area('Translation Text', trans_text_m2m_100, height=400, max_chars=1000,key='M2M_100')

        else:
            st.stop()

    elif process_box == 'URL':
        url_link = st.text_input('URL: https://example.com', 'http://www.slate.fr/story/207242/deliveroo-livraison-repas-mineurs-protohistoire-cereales',key=model_node)
        text_extract = ''
        try:
            # Assingn url
            #url = 'https://www.geeksforgeeks.org/top-5-open-source-online-machine-learning-environments/'
            # Extract web data
            url_i = newspaper.Article(url="%s" % (url_link))
            url_i.download()
            url_i.parse()
            # Display scrapped data
            text_extract = url_i.text
            print(text_extract)

        except:
            pass

        if len(text_extract) > 0:
            lang_detect = lang_detect_nlp(text_extract, nlp)
            _lang_detect_text_ = 'Detect Language: ' + lang_detect['language'].upper(), ' Score: {:.2f}'.format(
                lang_detect['score'])
            st.write(_lang_detect_text_,key='URL_lang_detect')
            if model_node == 'MBart_50':
                st.subheader('Model: ' + model_node.replace('_',''))
                target_MBart_50_URL = st.multiselect(label='target language', options=list(mbart_50_json.keys()),
                                            default='German', key='MBart_50_URL')
                mbart_sents_URL = nltk.tokenize.sent_tokenize(text_extract)  # don't use dlt.lang.ENGLISH
                trans_text_mbart_50_URL = " "
                if st.button('Translate', key='MBart_50_URL'):
                    fetch_list = query_fetch(set_lang_code(lang_detect['language']), 'mbart_50')
                    print('Fetch list: ' + str(fetch_list))
                    if len(fetch_list)>0:
                        source_MBart_50_URL = fetch_list[0]['language_name']
                        try:
                            trans_text_mbart_50_URL = " ".join(mbart_50_model_object.translate(mbart_sents_URL, source=source_MBart_50_URL,
                                                            target=target_MBart_50_URL[0], batch_size=32))
                        except:
                            trans_text_mbart_50_URL = "For this source language, we don't have support right now"
                    else:
                        trans_text_mbart_50_URL = "For this source language, we don't have support right now"
                    st.text_area('Translation Text', trans_text_mbart_50_URL, height=500,
                                                key='MBart_50_URL')
            elif model_node == 'M2M_100':
                st.subheader('Model: ' + model_node.replace('_',''))
                target_M2M_100_URL = st.multiselect(label='target language', options=list(mbart_50_json.keys()),
                                            default='English', key='M2M_100_URL')
                m2m_100_sents_URL = nltk.tokenize.sent_tokenize(text_extract)  # don't use dlt.lang.ENGLISH
                trans_text_m2m_100_URL = " "
                if st.button('Translate', key='M2M_100_URL'):
                    fetch_list_m2m_100 = query_fetch(set_lang_code(lang_detect['language']), 'm2m_100')
                    print('Fetch list: '+ str(fetch_list_m2m_100))
                    if len(fetch_list_m2m_100)>0:
                        source_M2M_100_URL = fetch_list_m2m_100[0]['language_name']
                        try:
                            trans_text_m2m_100_URL = " ".join(m2m_100_mode_object.translate(m2m_100_sents_URL, source=source_M2M_100_URL,
                                                            target=target_M2M_100_URL[0], batch_size=32))
                        except:
                            trans_text_m2m_100_URL = "For this source language, we don't have support right now"
                    else:
                        trans_text_m2m_100_URL = "For this source language, we don't have support right now"
                    st.text_area('Translation Text', trans_text_m2m_100_URL, height=500,
                                                key='M2M_100_URL')
            else:
                st.write('Select the translation model from sidebar.')
        else:
            st.write('Unable to extract text from URL')
    else:
        pass
about_mbart_500 = st.sidebar.checkbox('About mBART-50',key='about_mbart_50')
about_m2m_100 = st.sidebar.checkbox('About M2M100 418M',key='about_m2m_100')
if about_mbart_500:
    st.title('mBART-50 one to many multilingual machine translation')
    st.markdown("This model is a fine-tuned checkpoint of mBART-large-50."
                " **mbart-large-50-one-to-many-mmt** is fine-tuned for multilingual machine translation."
                " It was introduced in [**Multilingual Translation with Extensible Multilingual Pretraining"
                " and Finetuning **](https://arxiv.org/abs/2008.00401) paper")
    st.markdown("The model can translate English to other 49 languages mentioned below."
                " To translate into a target language, the target language id is forced as"
                " the first generated token. To force the target language id as the first generated"
                " token, pass the **forced_bos_token_id** parameter to the **generate** method.")
    st.markdown("See the [model hub](https://huggingface.co/models?filter=mbart-50) to look for more fine-tuned versions.")
    st.markdown("**Languages covered**")
    st.markdown("Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX),"
                " Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN),"
                " Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT),"
                " Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO),"
                " Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN),"
                " Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL),"
                " Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK),"
                " Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF),"
                " Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN),"
                " Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA),"
                " Galician (gl_ES), Slovene (sl_SI)")
    st.markdown('**BibTeX entry and citation info**')
    with st.echo():
        _article = 'tang2020multilingual',
        title = 'Multilingual Translation with Extensible Multilingual Pretraining and Finetuning',
        author = 'Yuqing Tang and Chau Tran and Xian Li and Peng-Jen Chen and Naman Goyal and Vishrav Chaudhary and Jiatao Gu and Angela Fan',
        year = 2020,
        eprint = 2008.00401,
        archivePrefix='arXiv',
        primaryClass='cs.CL'
if about_m2m_100:
    st.title('M2M100 418M')
    st.markdown('M2M100 is a multilingual encoder-decoder (seq-to-seq)'
                ' model trained for Many-to-Many multilingual translation.'
                ' It was introduced in this [paper](https://arxiv.org/abs/2010.11125) and'
                ' first released in [this](https://github.com/pytorch/fairseq/tree/master/examples/m2m_100) repository.')
    st.markdown('The model that can directly translate between the **9,900 directions of 100 languages**.'
                ' To translate into a target language, the target language id is forced as the first generated token.'
                ' To force the target language id as the first generated token,'
                ' pass the **forced_bos_token_id** parameter to the **generate** method.')
    st.markdown('Note: M2M100Tokenizer depends on sentencepiece')
    st.markdown('See the [model hub](https://huggingface.co/models?filter=m2m_100) to look for more fine-tuned versions.')
    st.markdown('**Languages covered**')
    st.markdown('Afrikaans (af), Amharic (am), Arabic (ar), Asturian (ast), Azerbaijani (az), Bashkir (ba),'
                ' Belarusian (be), Bulgarian (bg), Bengali (bn), Breton (br), Bosnian (bs), Catalan; Valencian (ca),'
                ' Cebuano (ceb), Czech (cs), Welsh (cy), Danish (da), German (de), Greeek (el), English (en), Spanish (es),'
                ' Estonian (et), Persian (fa), Fulah (ff), Finnish (fi), French (fr), Western Frisian (fy), Irish (ga),'
                ' Gaelic; Scottish Gaelic (gd), Galician (gl), Gujarati (gu), Hausa (ha), Hebrew (he), Hindi (hi),'
                ' Croatian (hr), Haitian; Haitian Creole (ht), Hungarian (hu), Armenian (hy), Indonesian (id), Igbo (ig),'
                ' Iloko (ilo), Icelandic (is), Italian (it), Japanese (ja), Javanese (jv), Georgian (ka), Kazakh (kk),'
                ' Central Khmer (km), Kannada (kn), Korean (ko), Luxembourgish; Letzeburgesch (lb), Ganda (lg), Lingala (ln),'
                ' Lao (lo), Lithuanian (lt), Latvian (lv), Malagasy (mg), Macedonian (mk), Malayalam (ml), Mongolian (mn),'
                ' Marathi (mr), Malay (ms), Burmese (my), Nepali (ne), Dutch; Flemish (nl), Norwegian (no), Northern Sotho (ns),'
                ' Occitan (post 1500) (oc), Oriya (or), Panjabi; Punjabi (pa), Polish (pl), Pushto; Pashto (ps), Portuguese (pt),'
                ' Romanian; Moldavian; Moldovan (ro), Russian (ru), Sindhi (sd), Sinhala; Sinhalese (si), Slovak (sk), Slovenian (sl),'
                ' Somali (so), Albanian (sq), Serbian (sr), Swati (ss), Sundanese (su), Swedish (sv), Swahili (sw), Tamil (ta),'
                ' Thai (th), Tagalog (tl), Tswana (tn), Turkish (tr), Ukrainian (uk), Urdu (ur), Uzbek (uz), Vietnamese (vi),'
                ' Wolof (wo), Xhosa (xh), Yiddish (yi), Yoruba (yo), Chinese (zh), Zulu (zu)')
    st.markdown('**BibTeX entry and citation info**')
    with st.echo():
        _misc = 'fan2020englishcentric',
        title = 'Beyond English-Centric Multilingual Machine Translation',
        author = 'Angela Fan and Shruti Bhosale and Holger Schwenk and Zhiyi Ma and Ahmed El-Kishky and Siddharth Goyal and Mandeep Baines and Onur Celebi and Guillaume Wenzek and Vishrav Chaudhary and Naman Goyal and Tom Birch and Vitaliy Liptchinsky and Sergey Edunov and Edouard Grave and Michael Auli and Armand Joulin',
        year = 2020,
        eprint = 2010.11125,
        archivePrefix='arXiv',
        primaryClass='cs.CL'