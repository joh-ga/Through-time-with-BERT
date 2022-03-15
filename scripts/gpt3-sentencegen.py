import os
import openai

#Use OpenAI Api for GPT-3
openai.api_key = os.getenv("OPENAI_API_KEY")
f = open("SentencesGPT.txt", "w")

#Set some parameters; best results (variance in verbs but not in personal pronome or sentence pattern) at temperature 0.7 and 800 tokens
for i in range(7):
    response = openai.Completion.create(
      engine="davinci",
      #give prompt for text generation (one shot setting)
      prompt="Present Simple Present: He goes to work. Present Continuous Tense: He is going to work. Present Perfect Tense: He has gone to work. Present Perfect Continuous Tense: He has been going to work. Simple Past Tense: He went to work.Past Continuous Tense: He was going to work. Past Perfect Tense: He had gone to work. Past Perfect Continuous Tense: He had been going to work. Simple Future Tense: He will go to work. Future Continuous Tense: He will be going to work. Future Perfect Tense: He will have gone to work. Future Perfect Continuous Tense: He is going to have been going to work. Present Simple Present: He drinks a glass of water.",
      temperature=0.7,
      max_tokens=800,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    f.write(str(response))
f.close()