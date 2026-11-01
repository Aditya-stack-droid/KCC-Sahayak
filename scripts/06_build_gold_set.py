"""Build the independent gold evaluation set (data/gold/gold_queries.csv).

150 hand-written farmer queries, 15 per QueryType, in three registers:
  en       natural conversational English (first person, not the terse
           "asking about X" style the KCC agents log)
  hinglish Hindi written in Roman script, as typed/spoken by many callers
  hi       Devanagari Hindi

Labels follow what the farmer is actually asking, NOT the keyword rules in
src/weak_labels.py -- so this set measures generalisation beyond the weak
labels. The author of this file wrote both queries and labels, which is
disclosed in the report; nothing here was sampled from the training corpus.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "gold" / "gold_queries.csv"

G = {}

G["Plant Protection"] = {
    "en": [
        "My wheat leaves have orange powdery spots all over and it is spreading fast to the neighbouring rows, which spray should I use?",
        "Tiny white insects fly up whenever I touch my cotton plants and the leaves are sticky, what is the remedy?",
        "The tomato plants in my polyhouse suddenly dried up from the base even though I water them regularly, is it some disease?",
        "Pink caterpillars are eating the bolls of my cotton, how do I stop them before the next picking?",
        "Brown patches with a yellow ring are showing up on potato leaves after the cloudy days, what should be done?",
    ],
    "hinglish": [
        "meri sarson ki fasal mein chepa lag gaya hai, patte chipchipe ho gaye hain, kaunsi dawai dalun",
        "gehu mein peela rust aa gaya hai kya spray karna chahiye",
        "dhan ke paudhe mein tana chedak ka hamla ho gaya hai, iska ilaaj batao",
        "mere aam ke ped par safed powder jaisa kuch aa raha hai aur phool jhad rahe hain",
        "kapas mein safed makkhi bahut ho gayi hai, kya karu",
    ],
    "hi": [
        "मेरी गेहूं की फसल में पत्तियां पीली पड़ रही हैं और उन पर जंग जैसे धब्बे हैं, कौन सी दवा डालूं?",
        "धान में तना छेदक कीट लग गया है, इसका उपचार बताइए",
        "टमाटर के पौधे मुरझा रहे हैं और जड़ के पास सड़न है, क्या करें?",
        "सरसों में माहू का प्रकोप बहुत ज्यादा है, कौन सा कीटनाशक छिड़कें?",
        "आलू की पत्तियों पर काले धब्बे पड़ गए हैं, झुलसा रोग से कैसे बचाएं?",
    ],
}

G["Nutrient Management"] = {
    "en": [
        "How much urea and DAP should I put per acre for wheat sowing in sandy loam soil?",
        "My paddy is pale green and growing slowly, do I need to give zinc or more nitrogen?",
        "Which fertiliser schedule should I follow for mustard if I only have compost and a little SSP?",
        "What is the correct dose of potash for sugarcane at the second irrigation stage?",
        "My sugarcane leaves show white stripes between the veins, is it a micronutrient deficiency and what should I apply?",
    ],
    "hinglish": [
        "ek bigha gehu mein kitna urea aur dap dalna chahiye",
        "dhan mein zinc ki kami ke liye kitna zinc sulphate daalna hai",
        "gobar ki khaad kitni matra mein daalun aloo ki buwai se pehle",
        "kapas ko pehli sinchai ke baad kaunsi khaad deni chahiye",
        "nitrogen kab aur kitna dena hai makke ki fasal mein",
    ],
    "hi": [
        "प्रति एकड़ गेहूं में यूरिया और डीएपी की कितनी मात्रा डालनी चाहिए?",
        "धान की पत्तियां पीली हैं, क्या जिंक की कमी है और कितना जिंक सल्फेट डालें?",
        "सरसों की फसल में सल्फर कब और कितना देना चाहिए?",
        "आलू में खाद की मात्रा कितनी रखनी चाहिए?",
        "गन्ने में पोटाश कब डालें?",
    ],
}

G["Weather"] = {
    "en": [
        "Is it going to rain in the next three days? I want to decide whether to harvest my wheat now.",
        "Will there be frost this week in my area? My potato crop is at a vulnerable stage.",
        "What is the temperature forecast for next week, I am planning to sow maize.",
        "A hailstorm warning has been issued, should I do anything to protect my standing crop?",
        "When will the monsoon arrive in Haryana this year, I am waiting to transplant paddy.",
    ],
    "hinglish": [
        "agle teen din mein baarish hogi kya, fasal kaatni hai",
        "is hafte paala padne ka khatra hai kya",
        "mausam kaisa rahega agle hafte, bijai karni hai",
        "aandhi aur olavrishti ki chetavani hai kya hamare ilake mein",
        "monsoon kab tak aayega is saal",
    ],
    "hi": [
        "अगले हफ्ते बारिश होने की संभावना है क्या?",
        "इस हफ्ते पाला पड़ने की चेतावनी है क्या, आलू की फसल लगी है",
        "कल मौसम कैसा रहेगा, कटाई करनी है",
        "आंधी और ओलावृष्टि का अलर्ट है क्या हमारे जिले में?",
        "इस साल मानसून कब आएगा?",
    ],
}

G["Government Scheme"] = {
    "en": [
        "How can I apply for a Kisan Credit Card and what documents does the bank need?",
        "I have not received my PM-Kisan instalment for two quarters, where do I check the status?",
        "Is there a subsidy available for installing drip irrigation on a two acre farm?",
        "My crop was destroyed by hail, how do I claim crop insurance under PMFBY?",
        "What loan schemes are available for buying a tractor and how much subsidy does the government give?",
    ],
    "hinglish": [
        "kisan credit card kaise banwaye, kaun se documents lagenge",
        "pm kisan ki kist nahi aayi, status kahan dekhun",
        "drip irrigation par subsidy kaise milegi",
        "fasal bima ka claim kaise karna hai, ole se fasal kharab ho gayi",
        "tractor ke liye loan aur subsidy ki yojana batao",
    ],
    "hi": [
        "किसान क्रेडिट कार्ड बनवाने के लिए क्या दस्तावेज चाहिए?",
        "पीएम किसान की किस्त नहीं आई, स्थिति कहां देखूं?",
        "ड्रिप सिंचाई पर सरकार से कितनी सब्सिडी मिलती है?",
        "फसल बीमा का दावा कैसे करें, ओलावृष्टि से फसल खराब हो गई",
        "ट्रैक्टर खरीदने के लिए कौन सी सरकारी ऋण योजना है?",
    ],
}

G["Market Price"] = {
    "en": [
        "What is the going rate for mustard in the nearest mandi today?",
        "Should I hold my wheat in storage or sell now, do you think prices will go up in the next few weeks?",
        "What is the MSP of paddy this year and where can I sell at that price?",
        "Which market gives the best price for onions in Haryana right now?",
        "Are cotton prices expected to fall after the new arrivals?",
    ],
    "hinglish": [
        "aaj mandi mein sarson ka bhav kya chal raha hai",
        "gehu abhi bechun ya rok kar rakhun, bhav badhega kya",
        "dhan ka msp kitna hai is saal",
        "pyaaz ka sabse accha rate kis mandi mein mil raha hai",
        "kapas ka bhav gir jayega kya",
    ],
    "hi": [
        "आज मंडी में सरसों का भाव क्या चल रहा है?",
        "गेहूं अभी बेचूं या रोक कर रखूं, भाव बढ़ेगा क्या?",
        "इस साल धान का न्यूनतम समर्थन मूल्य कितना है?",
        "प्याज का सबसे अच्छा भाव किस मंडी में मिल रहा है?",
        "कपास के दाम गिरने वाले हैं क्या?",
    ],
}

G["Varietal Information"] = {
    "en": [
        "Which wheat variety gives the highest yield for late sowing in December?",
        "Can you suggest a short duration paddy variety that matures before the winter?",
        "Where can I get certified seed of a hybrid tomato that resists leaf curl?",
        "Which mustard variety is best for a low rainfall area with salty soil?",
        "Tell me the name of a good bajra hybrid for the sandy areas of Haryana.",
    ],
    "hinglish": [
        "late buwai ke liye gehu ki kaun si kism sabse achi hai",
        "dhan ki kam samay mein pakne wali kism batao",
        "sarson ki kaun si variety khaare paani mein ho sakti hai",
        "tamatar ke acche beej kahan se milenge",
        "bajre ki sabse zyada paidawar wali kism kaun si hai",
    ],
    "hi": [
        "देर से बुवाई के लिए गेहूं की कौन सी किस्म सबसे अच्छी है?",
        "धान की कम समय में पकने वाली किस्म बताइए",
        "सरसों की कौन सी किस्म खारे पानी में हो सकती है?",
        "टमाटर का प्रमाणित बीज कहां से मिलेगा?",
        "बाजरे की सबसे ज्यादा पैदावार वाली किस्म कौन सी है?",
    ],
}

G["Animal Husbandry"] = {
    "en": [
        "My buffalo has reduced her milk after calving and is not eating well, what should I feed her?",
        "When should I vaccinate my cow against foot and mouth disease?",
        "My goat has had loose motions and has not eaten since yesterday, what treatment can I give?",
        "What is the right ration for dairy cows to increase milk fat?",
        "Hens in my poultry shed are dying suddenly with a swollen head, what could be the reason?",
    ],
    "hinglish": [
        "meri bhains ne doodh dena kam kar diya hai, kya khilaun",
        "gaay ko khurpka mouthpaka ka teeka kab lagwana chahiye",
        "bakri ko dast lag gaye hain, kya ilaaj karun",
        "pashuon ke liye hara chara kaun sa lagaun jo zyada doodh de",
        "murgiyon mein ranikhet bimari ka ilaaj kya hai",
    ],
    "hi": [
        "मेरी भैंस ने दूध कम कर दिया है, क्या खिलाऊं?",
        "गाय को खुरपका-मुंहपका का टीका कब लगवाना चाहिए?",
        "बकरी को दस्त लग गए हैं, क्या इलाज करें?",
        "ज्यादा दूध के लिए कौन सा हरा चारा बोना चाहिए?",
        "मुर्गियों में रानीखेत बीमारी का इलाज क्या है?",
    ],
}

G["Fisheries"] = {
    "en": [
        "How do I prepare a pond for fish farming and which species grow well in Haryana?",
        "My fish are floating at the surface every morning, is there an oxygen problem in the pond?",
        "What is the right stocking density for rohu and catla in a one acre pond?",
        "What feed should I give to carp fingerlings in the first month?",
        "White spots have appeared on my fish, how can I treat the pond water?",
    ],
    "hinglish": [
        "machhli palan ke liye talab kaise taiyar karein",
        "talab mein machhliyan subah ke samay upar aa rahi hain, kya karun",
        "rohu aur catla ke bachche kitne daalne chahiye ek acre mein",
        "machhli ko kya chara dena chahiye",
        "machhliyon mein safed daane ho gaye hain, dawai batao",
    ],
    "hi": [
        "मछली पालन के लिए तालाब कैसे तैयार करें?",
        "तालाब में सुबह मछलियां ऊपर आ रही हैं, क्या करें?",
        "एक एकड़ तालाब में रोहू और कतला के कितने बच्चे डालें?",
        "मछलियों को कौन सा दाना देना चाहिए?",
        "मछलियों पर सफेद दाग हो गए हैं, इलाज बताइए",
    ],
}

G["Cultural Practices"] = {
    "en": [
        "What is the best time and seed rate for sowing wheat after the paddy harvest?",
        "How many irrigations does mustard need and at what growth stages?",
        "What spacing should I keep when transplanting paddy seedlings?",
        "When should I prune my guava orchard and how much should I cut back?",
        "Can I grow a pulse as an intercrop in my sugarcane field, and how?",
    ],
    "hinglish": [
        "dhan katne ke baad gehu kab boun aur beej kitna lagega",
        "sarson mein kitni sinchai karni chahiye",
        "dhan ki ropai mein paudhe se paudhe ki doori kitni rakhein",
        "amrood ki kat-chhant kab karni chahiye",
        "ganne ke beech mein dal ki intercropping kaise karein",
    ],
    "hi": [
        "धान के बाद गेहूं की बुवाई का सही समय और बीज दर क्या है?",
        "सरसों में कितनी सिंचाई करनी चाहिए?",
        "धान की रोपाई में पौधे से पौधे की दूरी कितनी रखें?",
        "अमरूद के बाग की कटाई-छंटाई कब करें?",
        "गन्ने के बीच में दलहन की अंतर्वर्ती खेती कैसे करें?",
    ],
}

G["Other"] = {
    "en": [
        "Where is the nearest soil testing laboratory and how do I collect the sample?",
        "What moisture level should grain have before I put it in a storage bin?",
        "I want to start mushroom cultivation at home, how do I begin?",
        "How do I register with an agriculture university for a one week training course?",
        "Can you give me the phone number of the district agriculture office?",
    ],
    "hinglish": [
        "mitti ki jaanch kahan hoti hai aur sample kaise lena hai",
        "anaj ko bhandaran karne se pehle kitni nami honi chahiye",
        "ghar par mushroom ki kheti kaise shuru karun",
        "krishi vigyan kendra mein training ke liye kahan naam likhwaun",
        "zila krishi adhikari ka phone number chahiye",
    ],
    "hi": [
        "मिट्टी की जांच कहां होती है और नमूना कैसे लें?",
        "अनाज भंडारण से पहले उसमें कितनी नमी होनी चाहिए?",
        "घर पर मशरूम की खेती कैसे शुरू करें?",
        "कृषि विज्ञान केंद्र में प्रशिक्षण के लिए नाम कहां लिखवाएं?",
        "जिला कृषि अधिकारी का फोन नंबर चाहिए",
    ],
}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, styles in G.items():
        for style, texts in styles.items():
            for t in texts:
                rows.append({"text": t, "label": label, "style": style})
    for i, r in enumerate(rows):
        r["id"] = f"g{i:03d}"
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label", "style"], quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} gold queries to {OUT}")


if __name__ == "__main__":
    main()
