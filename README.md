Advanced Machine Learning term project (MET CS767, Boston University).

The question: can you predict whether HD closes up or down the next day, using price history, financial news, or both combined? I picked HD (Home Depot) because my last name is Hamra.

Short answer: sentiment is genuinely learnable (my fine-tuned BERT hits 0.99 ROC-AUC), but the next-day up/down direction is not predictable, not from price, not from news, not from both. Every direction model lands around 0.50, which is basically a coin flip. That's exactly what the efficient market hypothesis says should happen. The point of this project was to build the full pipeline and evaluate it honestly, not to pretend I beat the market.

Results
Task	Best model	Result	Verdict
Q1, price to direction	LSTM / GRU / CNN	~0.48 to 0.55 acc	random
Q2, stacked price ensemble	meta model (stacking)	~0.48 acc	random
Q3, news sentiment	GRU from scratch	0.918 ROC-AUC	strong
Q4, news sentiment	BERT fine-tuned	0.992 ROC-AUC, 0.96 acc	strong
Sentiment vs returns	LSTM / GRU / VADER / BERT	r ≈ 0	no signal
Headlines to direction	LSTM / GRU	0.646 acc (= baseline), AUC ≈ 0.45	no signal
Q5, fusion (price + news)	everything combined	0.489 ± 0.012 AUC (5 seeds)	random

The gap between the two strong rows and the rest is the whole finding: language is learnable, market direction isn't.

How to view it

The notebook is big and doesn't render on GitHub, so here are two ways to read it without downloading:

Open in Colab, run it end to end
Open in nbviewer, read only, renders instantly

The full reasoning behind each step is in Report - AdvML - Simon Hamra.pdf.

What I did

The project follows the 5 questions from the brief, each one building on the last.

Q1, price models. I pulled HD daily data from Yahoo Finance and added the indicators asked in the brief (moving averages, RSI, MACD, two volatility measures). One thing I realized early: the price grows a lot over the years, from about 20 to 400, so if I fed raw dollar values the model would basically just see the trend and the train/test split wouldn't make sense. So I turned most features into percentage changes and ratios so they stay comparable across time. Target is 1 if the next day closes higher, 0 if lower. I used the last 30 days as input and split by time (70/15/15), no shuffling, because order matters here. Then I built three models: LSTM, GRU and a 1D CNN.

Q2, stacking. I looked at bagging, boosting and stacking, and went with stacking: feed all three base model predictions into a meta model that learns to combine them. I kept it very simple (just a sigmoid) because the base models were already sitting around 0.54, basically random, so adding capacity just overfits.

Q3, news sentiment from scratch. I couldn't find a clean HD news dataset online, so I built my own: 2,640 headlines. I scraped MarketWatch for 2020 to 2026 and merged that with two cleaned Kaggle sources for 2010 to 2020, keeping only headlines that actually mention Home Depot. I trained an LSTM and a GRU on the Financial PhraseBank to label sentiment, handled its class imbalance (about 69% positive) with class weights so it doesn't just always predict positive, and used VADER as a rule-based baseline. Then the real test: does daily sentiment line up with HD's returns? It doesn't, all correlations are basically zero. Which makes sense, by the time a headline is out the price already moved.

Q4, Transformer and BERT. I built a Transformer from scratch (0.908 AUC, basically tied with the LSTM and GRU, because Transformers are data hungry and 1,400 sentences isn't enough) and fine-tuned BERT (3 epochs, 0.992 AUC). BERT clearly wins, and that's the whole point of transfer learning: it already understands language from pre-training, so a light fine-tune beats anything trained from zero on a small dataset. On the harder check, applying the models to my actual HD headlines instead of PhraseBank, BERT held at ~0.88 AUC while the from-scratch models dropped to ~0.70.

Q5, fusion. One final model that takes everything: the Q1 price features, the outputs of all the price models and the meta model, plus daily sentiment from every Q3/Q4 model and the headline count. The work here was lining it all up by date (price is one row per trading day, news is sparse, so I fill no-news days with neutral). Result: 0.489 ± 0.012 AUC over 5 seeds. Price-only and news-only versions land in the same place. Combining the two worlds didn't help, because there was no signal in either to begin with.

Tech stack

Python, TensorFlow / Keras 3, PyTorch (for BERT), HuggingFace Transformers, scikit-learn, yfinance, ta, vaderSentiment, pandas, NumPy.

Limitations (being honest)
Q5 uses in-sample price-model predictions as features for the earliest training years (those days were part of Q1 training), so that's a mild leakage in the fusion's training features. The test-period result isn't affected, but it's worth naming.
The HD headlines are scored by models trained on PhraseBank, not hand-labeled on HD text. My weak-label checks suggest the transfer is reasonable (especially BERT), but it's still an approximation.
The main result is a negative one by design. It's the correct answer for this task, but it means the value here is the pipeline and the honest evaluation, not the prediction.

Simon Hamra, github.com/simonhamra, simonhamra71@gmail.com
