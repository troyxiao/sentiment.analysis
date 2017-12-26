movieSentiment.py is the main code of the program. It’s a modified version based on https://github.com/Poyuli/sentiment.analysis.git

The modified part are the two dimensional reduction methods (PCA and Gini index). Gini index method did not really work, which was just part of my experiment. 

I also found that specifying a higher number of the parameter “num_dim” tends to give a better result.

The Performance Report is following:

VectorType 	Training Model	dim_reduce  	Scaling		Accuracy	n-component
TF-IDF	 	SVM 	 	chi2  		standard        0.87792		500
TF-IDF	 	SVM 	 	chi2		signed  	0.87524		500
TF-IDF	 	SVM 	 	chi2		unsigned    	0.87684		500
TF-IDF	 	SVM 	 	PCA		standard    	0.88676		500
TF-IDF	 	SVM 	 	PCA		standard    	0.88952		1200
TF-IDF	 	SVM 		PCA		signed    	0.88412		00
TF-IDF	 	SVM 	 	PCA		unsigned    	0.8864		500
Binary	 	SVM 	 	PCA		standard    	0.86972		500
Int	 	SVM 	 	PCA		standard    	0.86732		500
Word2vec	SVM 	 	PCA		standard   	0.87044		100


VectorType  Training Model  dim_reduce      Scaling                 Accuracy
TF-IDF      NB              chi2            no(scaling rise error)  0.87236
Binary      NB              chi2            no(scaling rise error)  0.86432
Int         NB              chi2            no(scaling rise error)  0.83704
Word2vec    NB              chi2            any scaling             rise error
TF-IDF      NB              PCA             any scaling             rise error
Binary      NB              PCA             any scaling             rise error
Int         NB              PCA             any scaling             rise error
Word2vec    NB              PCA             any scaling             rise error

raise ValueError("Input X must be non-negative”)    


VectorType  Training Model  dim_reduce      Scaling     OOB Score
TF-IDF      RF              chi2            no          0.8218
Binary      RF              chi2            no          0.82024
Int         RF              chi2            no          0.81772
TF-IDF      RF              PCA             no          0.79324
Binary      RF              PCA             no          0.7576
Int         RF              PCA             no          0.68696
Word2vec    RF              PCA             no          rise error
Word2vec    RF              chi2            no          rise error
                
TF-IDF      BT              chi2            no          0.81084
Int         BT              chi2            no          0.80004