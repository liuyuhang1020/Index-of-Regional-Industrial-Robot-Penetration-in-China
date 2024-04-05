# Index of Regional Industrial Robot Penetration in China

![Image text](banner/center_for_enterprise_research.png)

This project is affiliated with the [Center for Enterprise Research of Peking University](https://cer.gsm.pku.edu.cn)

- The main task of the research assistant is to identify industrial robot companies from all enterprises based on the text of their business scope, thereby establishing a database of Chinese industrial robot companies.

- Use MySQL to operate a remote machine database, filtering out 10 million manufacturing enterprises and 5,000 industrial robot manufacturing enterprises for use as classification data.

- During the process of NLP feature engineering, a "Business Scope Standardization" algorithm was designed. This algorithm can convert the extensive text of business scopes found in enterprise registration information into several parallel structured phrases, each containing one product or service offered by the enterprise, thereby improving the model's classification effect.

- The Word2Vec model was used to encode the segmented business scopes, and the XGBoost model was utilized for classification. The final model achieved an AUC score of 0.85.

