"""
MarketPulse — Source unique de vérité pour les Stopwords multilingues (Hyper-enrichie).
Couvre : Anglais, Français, Allemand, Arabe + Jours, Mois, Saisons, Interrogatifs, Bruit Web.
"""

MULTILINGUAL_STOPWORDS = {
    # ─────────────────────────────────────────────────────────
    # 1. ANGLAIS (Pronoms, Auxiliaires, Interrogatifs, Adverbes)
    # ─────────────────────────────────────────────────────────
    "the", "a", "an", "in", "on", "at", "for", "to", "with", "by", "is", "are", 
    "was", "were", "and", "or", "but", "of", "from", "its", "it", "as", "that", 
    "this", "they", "will", "says", "said", "can", "has", "have", "not", "no", 
    "yes", "do", "does", "did", "done", "be", "been", "being", "he", "she", 
    "we", "you", "they", "my", "your", "his", "her", "our", "their", "them", 
    "him", "me", "us", "who", "whom", "whose", "which", "what", "why", "how", 
    "where", "when", "all", "each", "every", "both", "few", "more", "most", 
    "other", "some", "such", "than", "too", "very", "just", "so", "than", 
    "too", "about", "above", "after", "again", "against", "all", "am", "an", 
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been", 
    "before", "being", "below", "between", "both", "but", "by", "can't", 
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", 
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", 
    "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", 
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", 
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", 
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", 
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", 
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", 
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", 
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", 
    "then", "there", "there's", "these", "they", "they'd", "they'll", 
    "they're", "they've", "this", "those", "through", "to", "too", "under", 
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", 
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where", 
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", 
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", 
    "you've", "your", "yours", "yourself", "yourselves",

    # ─────────────────────────────────────────────────────────
    # 2. FRANÇAIS (Articles, Pronoms, Prépositions, Conjonctions)
    # ─────────────────────────────────────────────────────────
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "en", "à", "au", 
    "aux", "que", "qui", "ce", "cette", "ces", "dans", "sur", "par", "pour", 
    "pas", "plus", "sont", "est", "ou", "mais", "ont", "fait", "ne", "se", 
    "ses", "son", "sa", "il", "elle", "ils", "elles", "nous", "vous", "tu", 
    "je", "mon", "ton", "notre", "votre", "leur", "leurs", "y", "en", "on", 
    "ou", "où", "donc", "or", "ni", "car", "si", "tout", "tous", "toute", 
    "toutes", "autre", "autres", "même", "mêmes", "comme", "dont", "quand", 
    "quel", "quelle", "quels", "quelles", "pourquoi", "comment", "combien", 
    "très", "trop", "peu", "moins", "aussi", "ainsi", "alors", "après", 
    "avant", "avec", "chez", "contre", "depuis", "derrière", "devant", 
    "durant", "entre", "envers", "malgré", "pendant", "sans", "sauf", 
    "selon", "sous", "vers", "voici", "voilà", "été", "être", "avoir", 
    "faire", "dire", "pouvoir", "vouloir", "savoir", "voir", "venir",

    # ─────────────────────────────────────────────────────────
    # 3. ALLEMAND (Articles, Pronoms, Adverbes, Conjonctions)
    # ─────────────────────────────────────────────────────────
    "der", "die", "das", "ein", "eine", "einen", "einem", "einer", "eines", "und", 
    "in", "im", "von", "zu", "den", "mit", "sich", "auf", "für", "ist", "nicht", 
    "nach", "wie", "als", "auch", "es", "an", "werden", "aus", "außer", "mehr", 
    "weniger", "sehr", "zu", "noch", "schon", "nur", "aber", "oder", "denn", 
    "weil", "dass", "wenn", "so", "sind", "war", "waren", "sein", "wird", 
    "wurde", "wurden", "haben", "hat", "hatte", "hatten", "ich", "du", "er", 
    "sie", "wir", "ihr", "mich", "dich", "ihn", "uns", "euch", "ihnen", "dem", 
    "des", "über", "unter", "durch", "gegen", "ohne", "um", "warum", "wie", 
    "was", "wer", "wo", "wann", "welche", "welcher", "welches", "dieser", 
    "diese", "dieses", "jeder", "jede", "jedes", "alle", "alles", "etwas", 
    "nichts", "kein", "keine", "keinen", "aber", "sondern", "während", 
    "bevor", "nachdem", "obwohl", "weil", "da", "damit", "ob", "wobei",

    # ─────────────────────────────────────────────────────────
    # 4. ARABE (Prépositions, Pronoms, Particules, Interrogatifs)
    # ─────────────────────────────────────────────────────────
    "في", "من", "على", "أن", "إلى", "عن", "هذا", "هذه", "التي", "الذي", 
    "ففي", "ولكن", "مع", "هل", "قد", "بل", "لا", "ما", "لم", "لن", 
    "هو", "هي", "هم", "هن", "أنت", "أنتم", "نحن", "وإلى", "وكما", "أو", 
    "أم", "إن", "أن", "كأن", "لكن", "ليت", "لعل", "إذا", "لو", "لماذا", 
    "كيف", "متى", "أين", "كم", "أي", "لان", "لأن", "جدا", "أكثر", "أقل", 
    "نعم", "ربما", "سوف", "تم", "كان", "كانت", "يكون", "تكون", "هؤلاء", 
    "ذلك", "تلك", "هكذا", "هنا", "هناك", "حيث", "الذين", "اللواتي", 
    "اللاتي", "مما", "به", "بها", "فيه", "فيهم", "منه", "منها", "عنه", 
    "عنها", "عليه", "عليها", "اليه", "إليه", "بين", "حتى", "منذ", 

    # ─────────────────────────────────────────────────────────
    # 5. TEMPOREL : JOURS (4 Langues)
    # ─────────────────────────────────────────────────────────
    # Anglais
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    # Français
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
    # Allemand
    "montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag",
    # Arabe
    "الإثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد",
    "الاثنين",

    # ─────────────────────────────────────────────────────────
    # 6. TEMPOREL : MOIS (4 Langues)
    # ─────────────────────────────────────────────────────────
    # Anglais
    "january", "february", "march", "april", "may", "june", 
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    # Français
    "janvier", "février", "mars", "avril", "mai", "juin", 
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
    # Allemand
    "januar", "februar", "märz", "april", "mai", "juni", 
    "juli", "august", "september", "oktober", "november", "dezember",
    # Arabe
    "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", 
    "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
    "جانفي", "فيفري", "مارس", "أفريل", "ماي", "جوان", "جويلييه", "أوت", "سنتبر", "نوفمبر", "دجنبر",

    # ─────────────────────────────────────────────────────────
    # 7. TEMPOREL : SAISONS & GÉNÉRIQUES (4 Langues)
    # ─────────────────────────────────────────────────────────
    # Saisons
    "spring", "summer", "autumn", "fall", "winter",
    "printemps", "été", "automne", "hiver",
    "frühling", "sommer", "herbst", "winter",
    "ربيع", "صيف", "خريف", "شتاء",
    # Génériques temporels
    "jour", "jours", "day", "days", "tag", "tage", "يوم", "أيام",
    "mois", "month", "months", "monat", "monate", "شهر", "شهور", "أشهر",
    "année", "year", "years", "jahr", "jahre", "سنة", "سنوات", "عام", "أعوام",
    "saison", "season", "seasons", "موسم", "مواسم",
    "aujourd", "hui", "today", "heute", "demain", "hier", "اليوم", "غدا", "أمس",
    "woche", "wochen", "week", "weeks", "semaine", "semaines", "أسبوع", "أسابيع",
    "stunde", "stunden", "hour", "hours", "heure", "heures", "ساعة", "ساعات",

    # ─────────────────────────────────────────────────────────
    # 8. BRUIT WEB, MARKETING & E-COMMERCE
    # ─────────────────────────────────────────────────────────
    "promo", "off", "save", "deals", "code", "codes", "discount",
    "annonce", "annonces", "ad", "ads", "advertisement", "إعلان", "إعلانات",
    "gratuit", "free", "premium", "sponsor", "sponsored", "مجاني", "مجانا", "مميز", "ممول", "برعاية",
    "abonnement", "subscribe", "newsletter", "cliquez", "click", "اشتراك", "اشترك", "نشرة", "إخبارية", "انقر", "اضغط",
    "good", "best", "better", "bad", "worst", "great",
    "bon", "meilleur", "pire", "bien", "très",
    "gut", "besser", "beste", "schlecht",
    "جيد", "أفضل", "أحسن", "سيء", "أسوأ", "عظيم", "ممتاز", "جدا"
}
