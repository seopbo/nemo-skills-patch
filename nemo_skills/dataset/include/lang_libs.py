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

from collections import namedtuple

MCQFormat = namedtuple("MCQFormat", ["q_label", "opt_label", "answer_prefix", "task"])

ANSWER_PLACEHOLDER = '{"answer":"X"}'
EXTRACT_REGEX = r'(?:\{"[^"]+"\s*:\s*"|[aA]nswer\s+is[\s*:-]+\(?)\s*([ABCD])\s*(?:\)?\.?\b|"\})'

MCQ_FORMATS = {
    "English": MCQFormat(
        q_label="Question:",
        opt_label="Options:",
        answer_prefix="Answer: Let's think step by step.",
        task='The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with {answer_placeholder} where X is the correct letter choice.',
    ),
    "Russian": MCQFormat(
        q_label="Вопрос:",
        opt_label="Варианты:",
        answer_prefix="Ответ: Давайте подумаем шаг за шагом.",
        task='Ниже приведен вопрос с множественным выбором о {subject} (с ответами). Пожалуйста, размышляйте шаг за шагом, а затем завершите свой ответ с {answer_placeholder}, где X - это буква правильного варианта.',
    ),
    "French": MCQFormat(
        q_label="Question:",
        opt_label="Options:",
        answer_prefix="Réponse: Réfléchissons étape par étape.",
        task='Voici des questions à choix multiples (avec réponses) sur {subject}. Réfléchissez étape par étape, puis terminez votre réponse par {answer_placeholder} où X est la lettre correspondant au bon choix.',
    ),
    "Spanish": MCQFormat(
        q_label="Pregunta:",
        opt_label="Opciones:",
        answer_prefix="Respuesta: Pensemos paso a paso.",
        task='Las siguientes son preguntas de opción múltiple (con respuestas) sobre {subject}. Piense paso a paso y luego termine su respuesta con {answer_placeholder} donde X es la letra de la opción correcta.',
    ),
    "German": MCQFormat(
        q_label="Frage:",
        opt_label="Optionen:",
        answer_prefix="Antwort: Denken wir Schritt für Schritt nach.",
        task='Im Folgenden sind Multiple-Choice-Fragen (mit Antworten) zu {subject}. Denken Sie Schritt für Schritt nach und beenden Sie Ihre Antwort mit {answer_placeholder}, wobei X der richtige Buchstabe ist.',
    ),
    "Portuguese": MCQFormat(
        q_label="Pergunta:",
        opt_label="Opções:",
        answer_prefix="Resposta: Vamos pensar passo a passo.",
        task='A seguir estão perguntas de múltipla escolha (com respostas) sobre {subject}. Pense passo a passo e termine sua resposta com {answer_placeholder} onde X é a letra da opção correta.',
    ),
    "Italian": MCQFormat(
        q_label="Domanda:",
        opt_label="Opzioni:",
        answer_prefix="Risposta: Ragioniamo passo dopo passo.",
        task='Ecco una domanda a scelta multipla su {subject} (con risposta). Si prega di ragionare passo dopo passo e terminare la risposta con {answer_placeholder}, dove X è la lettera dell\'opzione corretta.',
    ),
    "Serbian": MCQFormat(
        q_label="Pitanje:",
        opt_label="Opcije:",
        answer_prefix="Odgovor: Razmislimo korak po korak.",
        task='Evo pitanja sa višestrukim izborom o {subject} (sa odgovorom). Molimo vas da razmislite korak po korak i završite svoj odgovor sa {answer_placeholder}, gde je X slovo tačne opcije.',
    ),
    "Ukrainian": MCQFormat(
        q_label="Питання:",
        opt_label="Варіанти:",
        answer_prefix="Відповідь: Давайте подумаємо крок за кроком.",
        task='Ось запитання з вибором відповідей на тему {subject} (з відповіддю). Будь ласка, подумайте крок за кроком і закінчіть свою відповідь {answer_placeholder}, де X – літера правильного варіанту.',
    ),
    "Hungarian": MCQFormat(
        q_label="Kérdés:",
        opt_label="Opciók:",
        answer_prefix="Válasz: Gondolkodjunk lépésről lépésre.",
        task='Itt van egy feleletválasztós kérdés a(z) {subject} témában (választ is tartalmazza). Kérjük, gondolkodjon lépésről lépésre, és a válaszát a(z) {answer_placeholder} kifejezéssel fejezze be, ahol X a helyes válasz betűjele.',
    ),
    "Vietnamese": MCQFormat(
        q_label="Câu hỏi:",
        opt_label="Lựa chọn:",
        answer_prefix="Trả lời: Hãy suy nghĩ từng bước một.",
        task='Dưới đây là câu hỏi trắc nghiệm về {subject} (kèm đáp án). Vui lòng suy nghĩ từng bước, sau đó kết thúc câu trả lời của bạn bằng {answer_placeholder}, trong đó X là chữ cái của lựa chọn đúng.',
    ),
    "Urdu": MCQFormat(
        q_label="سوال:",
        opt_label="آپشنز:",
        answer_prefix="جواب: آئیے قدم بہ قدم سوچتے ہیں۔",
        task='درج ذیل {subject} کے متعلق ایک متعدد انتخابی سوال ہے (جوابات کے ساتھ)۔ براہ کرم قدم بہ قدم سوچیں، اور پھر اپنے جواب کو {answer_placeholder} کے ساتھ ختم کریں، جہاں X درست آپشن کا حرف ہے۔',
    ),
    "Indonesian": MCQFormat(
        q_label="Pertanyaan:",
        opt_label="Pilihan:",
        answer_prefix="Jawaban: Mari berpikir langkah demi langkah.",
        task='Berikut adalah pertanyaan pilihan ganda tentang {subject} (dengan jawaban). Harap berpikir langkah demi langkah, lalu akhiri jawaban Anda dengan {answer_placeholder}, di mana X adalah huruf pilihan yang benar.',
    ),
    "Korean": MCQFormat(
        q_label="질문：",
        opt_label="선택 사항：",
        answer_prefix="답변: 한 단계씩 생각해 봅시다.",
        task='다음은 {subject}에 관한 객관식 문제(정답 포함)입니다. 단계적으로 생각한 다음 {answer_placeholder}로 답변을 마무리하세요. 여기서 X는 올바른 선택지 문자입니다.',
    ),
    "Hindi": MCQFormat(
        q_label="प्रश्न:",
        opt_label="विकल्प:",
        answer_prefix="उत्तर: चलिए चरण-दर-चरण सोचते हैं।",
        task='निम्नलिखित {subject} के बारे में बहुविकल्पीय प्रश्न (उत्तरों के साथ) हैं। चरण-दर-चरण सोचें और फिर अपने उत्तर को {answer_placeholder} के साथ समाप्त करें जहां X सही विकल्प का अक्षर है।',
    ),
    "Arabic": MCQFormat(
        q_label="سؤال:",
        opt_label="الخيارات:",
        answer_prefix="الإجابة: دعنا نفكر خطوة بخطوة.",
        task='فيما يلي أسئلة اختيار من متعدد (مع إجابات) حول {subject}. فكر خطوة بخطوة ثم أنهِ إجابتك بـ {answer_placeholder} حيث X هو حرف الخيار الصحيح.',
    ),
    "Telugu": MCQFormat(
        q_label="ప్రశ్న:",
        opt_label="ఎంపికలు:",
        answer_prefix="సమాధానం: దశలవారీగా ఆలోచిద్దాం.",
        task='క్రింది {subject}కి సంబంధించిన బహుళఎంపిక ప్రశ్న (సమాధానాలతో). దయచేసి దశలవారీగా ఆలోచించి, మీ సమాధానాన్ని {answer_placeholder}తో ముగించండి, ఇక్కడ X సరైన ఎంపిక అక్షరం.',
    ),
    "Bengali": MCQFormat(
        q_label="প্রশ্ন:",
        opt_label="বিকল্পগুলি:",
        answer_prefix="উত্তর: আসুন ধাপে ধাপে চিন্তা করি।",
        task='নিম্নলিখিত {subject} সম্পর্কে বহুনির্বাচনী প্রশ্ন (উত্তরসহ)। ধাপে ধাপে চিন্তা করুন এবং তারপর আপনার উত্তর {answer_placeholder} দিয়ে শেষ করুন যেখানে X হল সঠিক বিকল্পের অক্ষর।',
    ),
    "Nepali": MCQFormat(
        q_label="प्रश्न:",
        opt_label="विकल्पहरू:",
        answer_prefix="उत्तर: चरणबद्ध रूपमा सोचौं।",
        task='यहाँ {subject} सम्बन्धी बहुवैकल्पिक प्रश्नहरू छन् (उत्तरहरू सहित)। कृपया चरणबद्ध रूपमा सोच्नुहोस् र आफ्नो उत्तर {answer_placeholder} बाट अन्त्य गर्नुहोस्, जहाँ X सही विकल्पको अक्षर हो।',
    ),
    "Japanese": MCQFormat(
        q_label="質問：",
        opt_label="選択肢：",
        answer_prefix="回答：一歩一歩考えていきましょう。",
        task="以下は{subject}に関する選択問題（解答付き）です。段階的に考え、最後に「{answer_placeholder}」と回答を締めくくってください。Xは正解の選択肢を示す文字です。",
    ),
    "Chinese": MCQFormat(
        q_label="问题：",
        opt_label="选项：",
        answer_prefix="答案：让我们一步一步地思考。",
        task='以下是关于{subject}的选择题（带有答案）。请逐步思考，然后以{answer_placeholder}结束您的回答，其中X是正确的选项字母。',
    ),
    "Albanian": MCQFormat(
        q_label="Pyetja:",
        opt_label="Opsionet:",
        answer_prefix="Përgjigje: Le ta mendojmë hap pas hapi.",
        task='Më poshtë janë pyetje me zgjedhje të shumëfishta (me përgjigje) rreth {subject}. Mendo hap pas hapi dhe pastaj përfundoje përgjigjen tënde me {answer_placeholder} ku X është varianti i saktë.',
    ),
    "Armenian": MCQFormat(
        q_label="Հարց:",
        opt_label="Տարբերակներ:",
        answer_prefix="Պատասխան․ Եկեք մտածենք քայլ առ քայլ։",
        task='Ստորև բերված են բազմընտրանի հարցեր (պատասխաններով) {subject}-ի վերաբերյալ։ Մտածեք քայլ առ քայլ, ապա ավարտեք ձեր պատասխանը {answer_placeholder}-ով, որտեղ X-ը ճիշտ տարբերակի տառն է։',
    ),
    "Azerbaijani": MCQFormat(
        q_label="Sual:",
        opt_label="Seçimlər:",
        answer_prefix="Cavab: Gəlin addım-addım düşünək.",
        task='Aşağıdakılar {subject} haqqında çoxseçimli suallardır (cavablarla). Addım-addım düşünün və cavabınızı {answer_placeholder} ilə bitirin, burada X düzgün variantın hərfidir.',
    ),
    "Basque": MCQFormat(
        q_label="Galdera:",
        opt_label="Aukerak:",
        answer_prefix="Erantzuna: Pentsa dezagun pausoz pauso.",
        task='{subject}-ri buruzko aukera anitzeko galderak dira hauek (erantzunekin). Pentsatu pausoz pauso eta amaitu zure erantzuna {answer_placeholder}-ekin, non X aukera zuzenaren letra den.',
    ),
    "Belarusian": MCQFormat(
        q_label="Пытанне:",
        opt_label="Варыянты:",
        answer_prefix="Адказ: Давайце падумаем крок за крокам.",
        task='Ніжэй прыведзены пытанні з варыянтамі адказаў (з адказамі) па тэме {subject}. Падумайце крок за крокам і завершце адказ фразай {answer_placeholder}, дзе X — літара правільнага варыянта.',
    ),
    "Bulgarian": MCQFormat(
        q_label="Въпрос:",
        opt_label="Опции:",
        answer_prefix="Отговор: Нека мислим стъпка по стъпка.",
        task='По-долу са въпроси с избираем отговор (с отговори) относно {subject}. Мислете стъпка по стъпка и завършете отговора си с {answer_placeholder}, където X е буквата на правилния избор.',
    ),
    "Croatian": MCQFormat(
        q_label="Pitanje:",
        opt_label="Opcije:",
        answer_prefix="Odgovor: Razmislimo korak po korak.",
        task='Slijede pitanja s višestrukim izborom (s odgovorima) o temi {subject}. Razmišljajte korak po korak i završite svoj odgovor s {answer_placeholder}, gdje je X slovo točnog odgovora.',
    ),
    "Dutch": MCQFormat(
        q_label="Vraag:",
        opt_label="Opties:",
        answer_prefix="Antwoord: Laten we stap voor stap nadenken.",
        task='Hieronder staan meerkeuzevragen (met antwoorden) over {subject}. Denk stap voor stap en eindig uw antwoord met {answer_placeholder}, waarbij X de letter van het juiste antwoord is.',
    ),
    "Estonian": MCQFormat(
        q_label="Küsimus:",
        opt_label="Valikud:",
        answer_prefix="Vastus: Mõtleme samm-sammult.",
        task='Järgnevad on valikvastustega küsimused (vastustega) teemal {subject}. Mõelge samm-sammult ja lõpetage oma vastus sõnadega {answer_placeholder}, kus X on õige vastuse täht.',
    ),
    "Finnish": MCQFormat(
        q_label="Kysymys:",
        opt_label="Vaihtoehdot:",
        answer_prefix="Vastaus: Mietitään askel askeleelta.",
        task='Seuraavat ovat monivalintakysymyksiä (vastauksineen) aiheesta {subject}. Mieti askel askeleelta ja päätä vastauksesi {answer_placeholder}-merkintään, jossa X on oikea vastausvaihtoehto.',
    ),
    "Georgian": MCQFormat(
        q_label="კითხვა:",
        opt_label="ვარიანტები:",
        answer_prefix="პასუხი: ვიფიქროთ ეტაპობრივად.",
        task='ქვემოთ წარმოდგენილია მრავალვარიანტიანი კითხვები (პასუხებით) {subject}-თან დაკავშირებით. იფიქრეთ ეტაპობრივად და დაასრულეთ თქვენი პასუხი {answer_placeholder}-ით, სადაც X სწორ ვარიანტს აღნიშნავს.',
    ),
    "Greek": MCQFormat(
        q_label="Ερώτηση:",
        opt_label="Επιλογές:",
        answer_prefix="Απάντηση: Ας σκεφτούμε βήμα-βήμα.",
        task='Οι παρακάτω είναι ερωτήσεις πολλαπλής επιλογής (με απαντήσεις) σχετικά με το {subject}. Σκεφτείτε βήμα-βήμα και ολοκληρώστε την απάντησή σας με {answer_placeholder}, όπου X είναι το σωστό γράμμα.',
    ),
    "Hebrew": MCQFormat(
        q_label="שאלה:",
        opt_label="אפשרויות:",
        answer_prefix="תשובה: בואו נחשוב צעד אחר צעד.",
        task='להלן שאלות אמריקאיות (עם תשובות) בנושא {subject}. חשבו צעד אחר צעד וסיימו את תשובתכם עם {answer_placeholder}, כאשר X הוא האות של התשובה הנכונה.',
    ),
    "Kazakh": MCQFormat(
        q_label="Сұрақ:",
        opt_label="Нұсқалар:",
        answer_prefix="Жауап: Қадам-қадам ойланайық.",
        task='{subject} бойынша көп таңдаулы сұрақтар (жауаптарымен) төменде берілген. Қадамдап ойланып, жауабыңызды {answer_placeholder} деп аяқтаңыз, мұнда X — дұрыс жауаптың әрпі.',
    ),
    "Lithuanian": MCQFormat(
        q_label="Klausimas:",
        opt_label="Pasirinkimai:",
        answer_prefix="Atsakymas: Pagalvokime žingsnis po žingsnio.",
        task='Toliau pateikiami kelių pasirinkimų klausimai (su atsakymais) apie {subject}. Mąstykite žingsnis po žingsnio ir užbaikite atsakymą fraze {answer_placeholder}, kur X yra teisingo varianto raidė.',
    ),
    "Malay": MCQFormat(
        q_label="Soalan:",
        opt_label="Pilihan:",
        answer_prefix="Jawapan: Mari kita fikir selangkah demi selangkah.",
        task='Berikut ialah soalan pilihan berganda (dengan jawapan) tentang {subject}. Fikir secara berperingkat dan tamatkan jawapan anda dengan {answer_placeholder}, di mana X ialah huruf pilihan yang betul.',
    ),
    "Malayalam": MCQFormat(
        q_label="ചോദ്യം:",
        opt_label="ഓപ്ഷനുകൾ:",
        answer_prefix="ഉത്തരം: നമുക്ക് ഘട്ടംഘട്ടമായി ചിന്തിക്കാം.",
        task='{subject} സംബന്ധിച്ച ബഹുവലിപ്പരീക്ഷാ ചോദ്യങ്ങളാണ് താഴെ കൊടുത്തിരിക്കുന്നത് (ഉത്തരങ്ങളോടു കൂടി). ഘട്ടംഘട്ടമായി ചിന്തിച്ച് നിങ്ങളുടെ ഉത്തരത്തിന്റെ അവസാനം {answer_placeholder} എന്ന് ചേർക്കുക, ഇവിടെ X ശരിയായ ഓപ്ഷന്റെ അക്ഷരമാണ്.',
    ),
    "North Macedonian": MCQFormat(
        q_label="Прашање:",
        opt_label="Опции:",
        answer_prefix="Одговор: Да размислуваме чекор по чекор.",
        task='Подолу се дадени прашања со повеќекратен избор (со одговори) за {subject}. Размислувајте чекор по чекор и завршете го одговорот со {answer_placeholder}, каде X е буквата на точниот избор.',
    ),
    "Persian": MCQFormat(
        q_label="سؤال:",
        opt_label="گزینه‌ها:",
        answer_prefix="پاسخ: بیایید گام به گام فکر کنیم.",
        task='در زیر پرسش‌های چندگزینه‌ای (با پاسخ) دربارهٔ {subject} آمده است. گام به گام فکر کنید و پاسخ خود را با {answer_placeholder} پایان دهید، جایی که X حرف گزینهٔ درست است.',
    ),
    "Polish": MCQFormat(
        q_label="Pytanie:",
        opt_label="Opcje:",
        answer_prefix="Odpowiedź: Zastanówmy się krok po kroku.",
        task='Poniżej znajdują się pytania wielokrotnego wyboru (z odpowiedziami) dotyczące {subject}. Myśl krok po kroku i zakończ odpowiedź frazą {answer_placeholder}, gdzie X oznacza poprawną literę odpowiedzi.',
    ),
    "Tagalog": MCQFormat(
        q_label="Tanong:",
        opt_label="Mga Opsyon:",
        answer_prefix="Sagot: Mag-isip tayo nang paisa-isang hakbang.",
        task='Ang mga sumusunod ay mga tanong na multiple choice (may mga sagot) tungkol sa {subject}. Mag-isip nang sunod-sunod at tapusin ang iyong sagot sa {answer_placeholder}, kung saan ang X ang tamang titik.',
    ),
    "Tamil": MCQFormat(
        q_label="கேள்வி:",
        opt_label="விருப்பங்கள்:",
        answer_prefix="பதில்: படிப்படியாக யோசிப்போம்.",
        task='{subject} பற்றிய பல்தேர்வு கேள்விகள் (பதில்களுடன்) கீழே கொடுக்கப்பட்டுள்ளன. படிப்படியாக யோசித்து, உங்கள் பதிலை {answer_placeholder} என முடிக்கவும், இதில் X சரியான தேர்வின் எழுத்தாகும்.',
    ),
    "Turkish": MCQFormat(
        q_label="Soru:",
        opt_label="Seçenekler:",
        answer_prefix="Cevap: Adım adım düşünelim.",
        task='{subject} hakkında çoktan seçmeli sorular (cevaplarıyla birlikte) aşağıda verilmiştir. Adım adım düşünün ve cevabınızı {answer_placeholder} ifadesiyle bitirin; burada X doğru seçeneğin harfidir.',
    ),
    "Uzbek": MCQFormat(
        q_label="Savol:",
        opt_label="Variantlar:",
        answer_prefix="Javob: Keling, bosqichma-bosqich o'ylaylik.",
        task='{subject} haqida ko‘p tanlovli savollar (javoblari bilan) quyida keltirilgan. Bosqichma-bosqich o‘ylang va javobingizni {answer_placeholder} bilan yakunlang, bu yerda X to‘g‘ri variant harfidir.',
    ),
}


def get_mcq_format(language, il_prompts):

    # In-Language Prompts
    if il_prompts:
        if language in MCQ_FORMATS:
            return MCQ_FORMATS[language]
        raise ValueError(f"Language {language} not supported")

    # English Prompts
    return MCQ_FORMATS["English"]


def supported_languages():
    # Exclude English;
    languages = sorted(list(MCQ_FORMATS.keys()))
    languages.remove("English")
    return languages
