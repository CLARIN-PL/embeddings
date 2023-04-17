import datasets
import pytest


@pytest.fixture(scope="module")
def sample_question_answering_dataset() -> datasets.DatasetDict:
    data = {
        "id": [
            "57262779271a42140099d5f1",
            "572fb38ea23a5019007fc8cb",
            "572953656aef051400154cf2",
            "5727951d708984140094e183",
            "56df787656340a1900b29bf8",
            "56ce0d8662d2951400fa69e8",
            "5a53e3bbbdaabd001a3867c0",
            "5706b9072eaba6190074ac7e",
            "5731309ce6313a140071cce8",
            "56cf69144df3c31400b0d745",
            "dKj4lOcFhJ7nSbW9Xq3eLmGx",
            "wNk6pRyUqH9sTgX8mL7fZdJc",
            "tGh2kMjNlP9cFeY3qR5sBwEa",
            "pWf8vZuNcD6mGtY5lHr9sQjK",
            "xLs9eTgPjK6cBmH8wZfQyUaD",
            "nMf2yQjK5sUcHgP8lRw6vBtZ",
            "tSd7vFqMkL9cXjR2wN4zGhYb",
            "dKj7sLpR6tZ9nTfQcX8vB2hY",
            "hDy2sWzJ6xL9pRc5nFq4vG8t",
            "cTb7vHsN6fZ2jLpK9rX5qG8m",
            "bRf2mX9zL8vK6tHs5nD7qPjW",
            "pRq8nKcV5sB2mLjH9zF6xG7t",
            "zDc6vHjK9bX2sRqP5fL8tW4n",
            "rJf5uNvB8dQcZtL6wHs4pKmW",
            "mPf9zGqK7rL5tYwNcD8vBjHs",
            "sRf2jHc5tZ8vB9nKpL6mXqWw",
            "nFq5tL7vH2sRjPbK9wX8cZm",
            "kHc8vL5sP6tRjN2mZ9wXqFb",
        ],
        "title": [
            "East_India_Company",
            "Transistor",
            "Bermuda",
            "Nonprofit_organization",
            "United_Nations_Population_Fund",
            "Frédéric_Chopin",
            "Armenians",
            "House_music",
            "Kievan_Rus%27",
            "Frédéric_Chopin",
            "The Importance of Proper Hydration",
            "The Benefits of Regular Exercise",
            "Tips for Making Healthy Eating Choices",
            "Benefits of Meditation for Mental Health",
            "Ways to Reduce Stress and Anxiety",
            "The Benefits of Spending Time Outdoors",
            "The Benefits of a Positive Attitude",
            "The Importance of Time Management",
            "The Benefits of Mindfulness Meditation",
            "The Importance of Positive Thinking",
            "The Importance of Self-Care",
            "The Importance of Financial Planning",
            "The Benefits of Yoga",
            "The Importance of Digital Privacy",
            "The Importance of Environmental Sustainability",
            "The Benefits of Reading",
            "The Benefits of Regular Exercise",
            "The Benefits of Volunteerism",
        ],
        "context": [
            "The company, which benefited from the imperial patronage, soon expanded its commercial trading "
            "operations, eclipsing the Portuguese Estado da Índia, which had established bases in Goa, Chittagong, "
            "and Bombay, which Portugal later ceded to England as part of the dowry of Catherine de Braganza. The "
            "East India Company also launched a joint attack with the Dutch United East India Company on Portuguese "
            "and Spanish ships off the coast of China, which helped secure their ports in China. The company "
            "established trading posts in Surat (1619), Madras (1639), Bombay (1668), and Calcutta (1690). By 1647, "
            "the company had 23 factories, each under the command of a factor or master merchant and governor if so "
            "chosen, and 90 employees in India. The major factories became the walled forts of Fort William in "
            "Bengal, Fort St George in Madras, and Bombay Castle.",
            "Because the electron mobility is higher than the hole mobility for all semiconductor materials, "
            "a given bipolar n–p–n transistor tends to be swifter than an equivalent p–n–p transistor. GaAs has the "
            "highest electron mobility of the three semiconductors. It is for this reason that GaAs is used in "
            "high-frequency applications. A relatively recent FET development, the high-electron-mobility transistor "
            "(HEMT), has a heterostructure (junction between different semiconductor materials) of aluminium gallium "
            "arsenide (AlGaAs)-gallium arsenide (GaAs) which has twice the electron mobility of a GaAs-metal barrier "
            "junction. Because of their high speed and low noise, HEMTs are used in satellite receivers working at "
            "frequencies around 12 GHz. HEMTs based on gallium nitride and aluminium gallium nitride (AlGaN/GaN "
            "HEMTs) provide a still higher electron mobility and are being developed for various applications.",
            'Once known as "the Gibraltar of the West" and "Fortress Bermuda", Bermuda today is defended by forces of '
            "the British government. For the first two centuries of settlement, the most potent armed force operating "
            "from Bermuda was its merchant shipping fleet, which turned to privateering at every opportunity. The "
            "Bermuda government maintained a local militia. After the American Revolutionary War, Bermuda was "
            "established as the Western Atlantic headquarters of the Royal Navy. Once the Royal Navy established a "
            "base and dockyard defended by regular soldiers, however, the militias were disbanded following the War "
            "of 1812. At the end of the 19th century, the colony raised volunteer units to form a reserve for the "
            "military garrison.",
            "Some NPOs which are particularly well known, often for the charitable or social nature of their "
            "activities performed during a long period of time, include Amnesty International, Oxfam, "
            "Rotary International, Kiwanis International, Carnegie Corporation of New York, Nourishing USA, "
            "DEMIRA Deutsche Minenräumer (German Mine Clearers), FIDH International Federation for Human Rights, "
            "Goodwill Industries, United Way, ACORN (now defunct), Habitat for Humanity, Teach For America, "
            "the Red Cross and Red Crescent organizations, UNESCO, IEEE, INCOSE, World Wide Fund for Nature, "
            "Heifer International, Translators Without Borders and SOS Children's Villages.",
            "In America, nonprofit organizations like Friends of UNFPA (formerly Americans for UNFPA) worked to "
            "compensate for the loss of United States federal funding by raising private donations.",
            "At the age of 21 he settled in Paris. Thereafter, during the last 18 years of his life, he gave only "
            "some 30 public performances, preferring the more intimate atmosphere of the salon. He supported himself "
            "by selling his compositions and teaching piano, for which he was in high demand. Chopin formed a "
            "friendship with Franz Liszt and was admired by many of his musical contemporaries, including Robert "
            "Schumann. In 1835 he obtained French citizenship. After a failed engagement to Maria Wodzińska, "
            "from 1837 to 1847 he maintained an often troubled relationship with the French writer George Sand. A "
            "brief and unhappy visit to Majorca with Sand in 1838–39 was one of his most productive periods of "
            "composition. In his last years, he was financially supported by his admirer Jane Stirling, "
            "who also arranged for him to visit Scotland in 1848. Through most of his life, Chopin suffered from poor "
            "health. He died in Paris in 1849, probably of tuberculosis.",
            "Armenian literature dates back to 400 AD, when Mesrop Mashtots first invented the Armenian alphabet. "
            "This period of time is often viewed as the Golden Age of Armenian literature. Early Armenian literature "
            'was written by the "father of Armenian history", Moses of Chorene, who authored The History of Armenia. '
            "The book covers the time-frame from the formation of the Armenian people to the fifth century AD. The "
            "nineteenth century beheld a great literary movement that was to give rise to modern Armenian literature. "
            "This period of time, during which Armenian culture flourished, is known as the Revival period (Zartonki "
            "sherchan). The Revivalist authors of Constantinople and Tiflis, almost identical to the Romanticists of "
            "Europe, were interested in encouraging Armenian nationalism. Most of them adopted the newly created "
            "Eastern or Western variants of the Armenian language depending on the targeted audience, and preferred "
            "them over classical Armenian (grabar). This period ended after the Hamidian massacres, when Armenians "
            "experienced turbulent times. As Armenian history of the 1920s and of the Genocide came to be more openly "
            "discussed, writers like Paruyr Sevak, Gevork Emin, Silva Kaputikyan and Hovhannes Shiraz began a new era "
            "of literature.",
            "Towards the end of the 1990s and into the 2000s, producers such as Daft Punk, Stardust, Cassius, "
            "St. Germain and DJ Falcon began producing a new sound out of Paris's house scene. Together, "
            "they laid the groundwork for what would be known as the French house movement. By combining the "
            "harder-edged-yet-soulful philosophy of Chicago house with the melodies of obscure funk, state-of-the-art "
            "production techniques and the sound of analog synthesizers, they began to create the standards that "
            "would shape all house music.",
            "Vladimir's choice of Eastern Christianity may also have reflected his close personal ties with "
            "Constantinople, which dominated the Black Sea and hence trade on Kiev's most vital commercial route, "
            "the Dnieper River. Adherence to the Eastern Church had long-range political, cultural, and religious "
            "consequences. The church had a liturgy written in Cyrillic and a corpus of translations from Greek that "
            "had been produced for the Slavic peoples. This literature facilitated the conversion to Christianity of "
            "the Eastern Slavs and introduced them to rudimentary Greek philosophy, science, and historiography "
            "without the necessity of learning Greek (there were some merchants who did business with Greeks and "
            "likely had an understanding of contemporary business Greek). In contrast, educated people in medieval "
            "Western and Central Europe learned Latin. Enjoying independence from the Roman authority and free from "
            "tenets of Latin learning, the East Slavs developed their own literature and fine arts, quite distinct "
            "from those of other Eastern Orthodox countries.[citation needed] (See Old East Slavic language and "
            "Architecture of Kievan Rus for details ). Following the Great Schism of 1054, the Rus' church maintained "
            "communion with both Rome and Constantinople for some time, but along with most of the Eastern churches "
            "it eventually split to follow the Eastern Orthodox. That being said, unlike other parts of the Greek "
            "world, Kievan Rus' did not have a strong hostility to the Western world.",
            "In September 1828 Chopin, while still a student, visited Berlin with a family friend, zoologist Feliks "
            "Jarocki, enjoying operas directed by Gaspare Spontini and attending concerts by Carl Friedrich Zelter, "
            "Felix Mendelssohn and other celebrities. On an 1829 return trip to Berlin, he was a guest of Prince "
            "Antoni Radziwiłł, governor of the Grand Duchy of Posen—himself an accomplished composer and aspiring "
            "cellist. For the prince and his pianist daughter Wanda, he composed his Introduction and Polonaise "
            "brillante in C major for cello and piano, Op. 3.",
            "Drinking enough water is crucial for maintaining good health, as it helps regulate body temperature, "
            "keep joints lubricated, and remove waste. Without proper hydration, our bodies can become dehydrated,"
            " leading to fatigue, dizziness, and other symptoms.",
            "Regular exercise has numerous benefits for both physical and mental health. It can help reduce the risk "
            "of chronic diseases like heart disease and diabetes, improve mood and mental health, and increase energy "
            "levels and overall well-being.",
            "Making healthy eating choices can be a challenge, especially when surrounded by tempting but unhealthy "
            "options. To make it easier, try planning out your meals in advance, focusing on whole, nutrient-dense "
            "foods, and allowing yourself occasional treats in moderation.",
            "Meditation has been shown to have numerous benefits for mental health. It can help reduce symptoms of "
            "anxiety and depression, improve focus and concentration, and increase feelings of well-being and "
            "happiness.",
            "Stress and anxiety can take a toll on both physical and mental health. To reduce these feelings, "
            "try practicing relaxation techniques like deep breathing or yoga, getting regular exercise, and seeking "
            "support from friends, family, or a mental health professional.",
            "Spending time outdoors can have numerous benefits for physical and mental health. It can help reduce "
            "stress and anxiety, improve mood and cognitive function, and increase exposure to natural light and "
            "fresh air.",
            "A positive attitude can have numerous benefits for mental and physical health. It can help reduce stress "
            "and anxiety, improve relationships with others, and increase resilience in the face of challenges.",
            "Effective time management is key to achieving success in both personal and professional life. By "
            "prioritizing tasks, setting goals, and minimizing distractions, individuals can increase productivity "
            "and reduce stress.",
            "Mindfulness meditation has been shown to have numerous benefits for mental health. It can help reduce "
            "stress and anxiety, improve focus and concentration, and promote feelings of calm and relaxation.",
            "Positive thinking can have a powerful impact on mental health and well-being. By focusing on the good in "
            "life and practicing gratitude, individuals can reduce negative thoughts and improve overall mood.",
            "Self-care is crucial for maintaining good mental and physical health. By taking time for oneself and "
            "engaging in activities that promote relaxation and well-being, individuals can reduce stress and improve "
            "overall quality of life.",
            "Effective financial planning is key to achieving long-term financial goals. By creating a budget, "
            "saving money, and investing wisely, individuals can secure their financial future and reduce financial "
            "stress.",
            "Yoga has been shown to have numerous physical and mental health benefits. It can help reduce stress and "
            "anxiety, improve flexibility and balance, and promote feelings of calm and relaxation.",
            "Protecting one's digital privacy is crucial in today's connected world. By using strong passwords, "
            "avoiding public Wi-Fi, and being mindful of personal information shared online, individuals can reduce "
            "the risk of identity theft and other cybercrimes.",
            "Protecting the environment is crucial for the health of the planet and future generations. By reducing "
            "waste, conserving energy, and using environmentally friendly products, individuals can help promote "
            "sustainability and reduce the impact of climate change.",
            "Reading has numerous benefits for mental health and cognitive function. It can improve vocabulary and "
            "language skills, reduce stress and anxiety, and promote empathy and understanding.",
            "Regular exercise has numerous benefits for physical and mental health. It can improve cardiovascular "
            "health, reduce the risk of chronic diseases, and promote feelings of happiness and well-being.",
            "Volunteerism has numerous benefits for mental health and social connections. It can improve feelings of "
            "purpose and meaning, reduce social isolation, and promote a sense of community and belonging.",
        ],
        "question": [
            "what were the walled forts of Fort William in Bengal, Fort St George in Madras and Bombay castle before "
            "they were forts?",
            "What are common applications of HEMT?",
            "Due to the British goverment's defense forces, what are two nicknames for bermuda?",
            "What is a well known NPO that helps people from low incomes become homeowners?",
            "What is one country in which nonprofit organizations try to make up for the loss of United States "
            "funding for the UNFPA?",
            "Where did he end up living when he was 21?",
            "What was Zartonki Sherchan interested in openly discussing?",
            "Daft Punk began producing a new sound out of what european city?",
            "What was considered too be Kiev's most important route for trade?",
            "When did Chopin visit Berlin?",
            "Why is drinking enough water crucial for maintaining good health?",
            "What are some benefits of regular exercise?",
            "What are some tips for making healthy eating choices?",
            "What are some benefits of meditation for mental health?",
            "What are some ways to reduce stress and anxiety?",
            "What are some benefits of spending time outdoors?",
            "What are some benefits of a positive attitude?",
            "Why is time management important?",
            "What are some benefits of mindfulness meditation?",
            "Why is positive thinking important?",
            "Why is self-care important?",
            "Why is financial planning important?",
            "What are some benefits of practicing yoga?",
            "Why is digital privacy important?",
            "Why is environmental sustainability important?",
            "What are some benefits of reading?",
            "What are some benefits of regular exercise?",
            "What are some benefits of volunteerism?",
        ],
        "answers": [
            {"text": ["major factories"], "answer_start": [742]},
            {"text": ["satellite receivers"], "answer_start": [680]},
            {"text": ['the Gibraltar of the West" and "Fortress Bermuda"'], "answer_start": [15]},
            {"text": ["Habitat for Humanity"], "answer_start": [434]},
            {"text": ["America"], "answer_start": [3]},
            {"text": ["Paris"], "answer_start": [31]},
            {"text": [], "answer_start": []},
            {"text": ["Paris"], "answer_start": [158]},
            {"text": ["Dnieper River"], "answer_start": [200]},
            {"text": ["September 1828"], "answer_start": [3]},
            {"text": ["it helps regulate body temperature"], "answer_start": [65]},
            {"text": ["reduce the risk of chronic diseases"], "answer_start": [88]},
            {"text": ["focusing on whole, nutrient-dense foods"], "answer_start": [171]},
            {"text": ["reduce symptoms of anxiety and depression"], "answer_start": [83]},
            {"text": ["practicing relaxation techniques"], "answer_start": [101]},
            {"text": ["reduce stress and anxiety"], "answer_start": [94]},
            {"text": ["reduce stress and anxiety"], "answer_start": [91]},
            {"text": ["increase productivity and reduce stress"], "answer_start": [177]},
            {"text": ["reduce stress and anxiety"], "answer_start": [95]},
            {"text": ["reduce negative thoughts and improve overall mood"], "answer_start": [152]},
            {"text": ["reduce stress and improve overall quality of life"], "answer_start": [180]},
            {
                "text": ["secure their financial future and reduce financial stress"],
                "answer_start": [150],
            },
            {"text": ["reduce stress and anxiety"], "answer_start": [86]},
            {
                "text": ["reduce the risk of identity theft and other cybercrimes"],
                "answer_start": [195],
            },
            {
                "text": ["promote sustainability and reduce the impact of climate change"],
                "answer_start": [195],
            },
            {"text": ["improve vocabulary and language skills"], "answer_start": [79]},
            {"text": ["improve cardiovascular health"], "answer_start": [78]},
            {"text": ["improve feelings of purpose and meaning"], "answer_start": [84]},
        ],
    }
    dataset = datasets.Dataset.from_dict(data)
    return datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(dataset[0:14]),
            "validation": datasets.Dataset.from_dict(dataset[14:19]),
            "test": datasets.Dataset.from_dict(dataset[19:]),
        }
    )
