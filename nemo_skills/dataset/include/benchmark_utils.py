# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from datasets import load_dataset
from collections import defaultdict


# Dataset schema defined in Hugging Face datasets
class Schema:
    ANSWER: str = "answer"
    LANGUAGE: str = "language"
    DOMAIN: str = "domain"
    QUESTION: str = "question"
    SUBJECT: str = "subject"
    COUNTRY: str = "country"
    REGIONAL_FEATURE: str = "regional_feature"
    LEVEL: str = "level"
    OPTIONS: list[str] = [
        "option_a",
        "option_b",
        "option_c",
        "option_d",
    ]  # `option_{x}` fields are available only for base subset
    CHOICES: str = "choices"  # `choices` field is available only for lite subset


def load_include_datasets(languages, subset, split):
    return [
        load_dataset(f"CohereLabs/include-{subset}-44", lang)[split]
        for lang in languages
    ]


def load_few_shot_split(languages, few_shot_split="validation"):
    val_datasets = load_include_datasets(languages, "base", few_shot_split)
    few_shot_examples = {}
    for dataset, lang in zip(val_datasets, languages):
        subject_dict = defaultdict(list)
        for entry in dataset:
            subject_dict[entry[Schema.SUBJECT]].append(entry)
        few_shot_examples[lang] = subject_dict
    return few_shot_examples


def retrieve_few_shot_examples(few_shot_examples, language, subject, num_fewshot):
    retrieved_examples = []

    # If the language is not in the few-shot examples, return an empty list
    if language not in few_shot_examples:
        return retrieved_examples

    # Prefer the subject-specific few-shot examples
    if subject in few_shot_examples[language]:
        retrieved_examples.extend(few_shot_examples[language][subject][:num_fewshot])

    # If we still need more examples, use the other subjects
    if len(retrieved_examples) < num_fewshot:
        for s in few_shot_examples[language]:
            if s != subject:
                retrieved_examples.append(few_shot_examples[language][s][0])

            if len(retrieved_examples) >= num_fewshot:
                break

    # If we still need more examples, print a warning
    if len(retrieved_examples) < num_fewshot:
        print(
            f"Warning: Only {len(retrieved_examples)} few-shot examples found for {subject} in {language}"
            "Try decreasing the number of few-shot examples."
        )
    return retrieved_examples


QUESTION_TEMPLATE = (
    "{question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nAnswer: "
)
FEWSHOT_DELIMITER = "\n\n"
ENG_COT_PREFIX = "Let's think step by step."
ENG_ZERO_SHOT_DESCRIPTION = "The following is multiple-choice question about {subject}. Respond with the letter of the correct answer."
ENG_FEWSHOT_DESCRIPTION = (
    "The following are multiple-choice questions (with answers) about {subject}. Respond with the letter of the correct answer."
)
LATTER_REGEX = r"\b\(?\s*([ABCD])\s*\)?\.?\b"
EXTRACT_REGEX = r"[\s\S]*" + LATTER_REGEX

DESCRIPTION_TEMPLATES = {
    "Albanian": "Më poshtë janë pyetjet me zgjedhje të shumëfishta (me përgjigje) rreth {subject}.",
    "Arabic": "فيما يلي أسئلة اختيارية متعددة (مع الإجابات) حول {subject}.",
    "Armenian": "Ստորև բերված են բազմակի ընտրության հարցեր (պատասխաններով) {subject}-ի վերաբերյալ:",
    "Azerbaijani": "Aşağıdakılar {subject} haqqında çoxseçimli suallardır (cavabları ilə).",
    "Basque": "Honako hauek aukera anitzeko galderak dira (erantzunekin) {subject}-i buruz.",
    "Belarusian": "Ніжэй прыведзены пытанні з некалькімі варыянтамі адказаў (з адказамі) пра {subject}.",
    "Bengali": "নিম্নলিখিতগুলি {subject} সম্পর্কে বহু-পছন্দের প্রশ্ন (উত্তর সহ)।",
    "Bulgarian": "Следват въпроси с избираем отговор (с отговори) за {subject}.",
    "Chinese": "以下是关于 {subject} 的多项选择题（附答案）。",
    "Croatian": "Slijede pitanja s višestrukim izborom (s odgovorima) o {subject}.",
    "Dutch": "Hieronder staan ​​meerkeuzevragen (met antwoorden) over {subject}.",
    "Estonian": "Järgmised on valikvastustega küsimused (koos vastustega) teemal {subject}.",
    "Finnish": "Seuraavat ovat monivalintakysymyksiä (vastauksineen) aiheesta {subject}.",
    "French": "Les questions suivantes sont à choix multiples (avec réponses) sur {subject}.",
    "Georgian": "ქვემოთ მოცემულია რამდენიმე არჩევანის კითხვები (პასუხებით) {subject}-ის შესახებ.",
    "German": "Nachfolgend finden Sie Multiple-Choice-Fragen (mit Antworten) zu {subject}.",
    "Greek": "Ακολουθούν ερωτήσεις πολλαπλής επιλογής (με απαντήσεις) σχετικά με το {subject}.",
    "Hebrew": "להלן שאלות ברירות רבות (עם תשובות) על {subject}.",
    "Hindi": "निम्नलिखित {subject} के बारे में बहुविकल्पीय प्रश्न (उत्तर सहित) हैं।",
    "Hungarian": "'Az alábbiak feleletválasztós kérdések (válaszokkal) a következővel kapcsolatban: {subject}.'",
    "Indonesian": "Berikut ini adalah pertanyaan pilihan ganda (dengan jawaban) tentang {subject}.",
    "Italian": "Di seguito sono riportate domande a scelta multipla (con risposte) su {subject}.",
    "Japanese": "以下は、{subject} に関する複数選択の質問（回答付き）です。",
    "Kazakh": "Төменде {subject} туралы бірнеше таңдаулы сұрақтар (жауаптары бар) берілген.",
    "Korean": "다음은 {subject}에 대한 객관식 질문(답변 포함)입니다.",
    "Lithuanian": "Toliau pateikiami klausimai su atsakymų variantais (su atsakymais) apie {subject}.",
    "Malay": "Berikut ialah soalan aneka pilihan (dengan jawapan) tentang {subject}.",
    "Malayalam": "ഇനിപ്പറയുന്നവ {subject} നെക്കുറിച്ചുള്ള മൾട്ടിപ്പിൾ ചോയ്‌സ് ചോദ്യങ്ങളാണ് (ഉത്തരങ്ങളോടെ).",
    "Nepali": "निम्न {subject} को बारेमा बहु-छनौट प्रश्नहरू (उत्तरहरू सहित) छन्।",
    "North Macedonian": "Следниве се прашања со повеќекратен избор (со одговори) за {subject}.",
    "Persian": "در زیر سؤالات چند گزینه ای (همراه با پاسخ) در مورد {subject} آمده است.",
    "Polish": "Poniżej znajdują się pytania wielokrotnego wyboru (z odpowiedziami) na temat {subject}.",
    "Portuguese": "A seguir estão perguntas de múltipla escolha (com respostas) sobre {subject}.",
    "Russian": "Ниже приведены вопросы с несколькими вариантами ответов (с ответами) по теме {subject}.",
    "Serbian": "Следе питања са вишеструким избором (са одговорима) о {subject}.",
    "Spanish": "Las siguientes son preguntas de opción múltiple (con respuestas) sobre {subject}.",
    "Tagalog": "Ang mga sumusunod ay maramihang pagpipiliang tanong (na may mga sagot) tungkol sa {subject}.",
    "Tamil": "பின்வருபவை {subject} பற்றிய பல தேர்வு கேள்விகள் (பதில்களுடன்) உள்ளன.",
    "Telugu": "{subject} గురించిన బహుళ-ఎంపిక ప్రశ్నలు (సమాధానాలతో) క్రిందివి.",
    "Turkish": "Aşağıda {subject} ile ilgili çoktan seçmeli sorular (cevaplarıyla birlikte) bulunmaktadır.",
    "Ukrainian": "Нижче наведено запитання з кількома варіантами відповідей (з відповідями) про {subject}.",
    "Urdu": "ذیل میں {subject} کے بارے میں متعدد انتخابی سوالات (جوابات کے ساتھ) ہیں۔",
    "Uzbek": "Quyida {subject} boʻyicha koʻp tanlovli savollar (javoblari bilan) keltirilgan.",
    "Vietnamese": "Sau đây là các câu hỏi trắc nghiệm (có đáp án) về {subject}.",
}

SUPPORTED_LANGUAGES = sorted(list(DESCRIPTION_TEMPLATES.keys()))


def digit_to_letter(digit):
    return chr(ord("A") + int(digit))


def normalize_entry_field(entry, key):
    return (entry.get(key, "") or "").replace(" ", "_")


def copy_other_fields(entry):
    return {
        k: normalize_entry_field(entry, k)
        for k in [
            Schema.COUNTRY,
            Schema.REGIONAL_FEATURE,
            Schema.LEVEL,
            Schema.SUBJECT,
            Schema.LANGUAGE,
            Schema.DOMAIN,
        ]
    }


def get_mcq_fields(
    target_question,
    target_options,
    language,
    subject,
    il_prompts,
    num_fewshot,
    few_shot_examples,
):
    target_options_dict = {
        digit_to_letter(i): option for i, option in enumerate(target_options)
    }
    target_options_text = "\n".join(
        f"{letter}. {option}" for letter, option in target_options_dict.items()
    )

    eng_prompt = il_prompts == False
    if num_fewshot == 0:
        if eng_prompt:
            prompt = ENG_ZERO_SHOT_DESCRIPTION.format(subject=subject)
        else:
            prompt = DESCRIPTION_TEMPLATES[language].format(subject=subject)
        prompt += FEWSHOT_DELIMITER
        prompt += QUESTION_TEMPLATE.format(
            question=target_question,
            option_a=target_options_dict["A"],
            option_b=target_options_dict["B"],
            option_c=target_options_dict["C"],
            option_d=target_options_dict["D"],
        )
        if eng_prompt:
            prompt += ENG_COT_PREFIX
    else:
        shots = retrieve_few_shot_examples(
            few_shot_examples, language, subject, num_fewshot
        )
        shot_answers = [digit_to_letter(shot[Schema.ANSWER]) for shot in shots]
        shots = [
            QUESTION_TEMPLATE.format(
                question=shot[Schema.QUESTION],
                option_a=shot[Schema.OPTIONS[0]],
                option_b=shot[Schema.OPTIONS[1]],
                option_c=shot[Schema.OPTIONS[2]],
                option_d=shot[Schema.OPTIONS[3]],
            )
            + answer
            for shot, answer in zip(shots, shot_answers)
        ]
        if eng_prompt:
            prompt = ENG_FEWSHOT_DESCRIPTION.format(subject=subject)
        else:
            prompt = DESCRIPTION_TEMPLATES[language].format(subject=subject)
        prompt += FEWSHOT_DELIMITER
        prompt += FEWSHOT_DELIMITER.join(shots)
        prompt += FEWSHOT_DELIMITER
        prompt += QUESTION_TEMPLATE.format(
            question=target_question,
            option_a=target_options_dict["A"],
            option_b=target_options_dict["B"],
            option_c=target_options_dict["C"],
            option_d=target_options_dict["D"],
        )
    return {"question": prompt, "options": target_options_text, **target_options_dict}
