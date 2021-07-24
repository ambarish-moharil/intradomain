import pandas as pd
import re 
import requests
import os
import nltk
import string 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize
import torch
from transformers import BertTokenizer, BertModel
from transformers import BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt
import shutil
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.spatial.distance import cosine
import pickle 
from sklearn_extra.cluster import KMedoids
import os


def disambiguate():


    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states = True, # Whether the model returns all hidden-states.
                                      )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



    # Cleaning the text
    def standardize_text(text_field):
        text_field = text_field.replace(r"http\S+", " ")
        text_field = text_field.replace(r"http", " ")
        text_field = text_field.replace(r"(\d)", " ")
        text_field = text_field.replace(r"@\S+", " ")
        text_field = text_field.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n,\\,/,.,:,;,'""']", " ")
        text_field = text_field.replace(r"\\", " ")
        text_field = text_field.replace(r".", " ")
        text_field = text_field.replace(r";", " ")
        text_field = text_field.replace(r",", " ")
        text_field = text_field.replace(r":", " ")
        text_field = text_field.replace(r"←", " ")
        text_field = text_field.replace(r"≠", " ")
        text_field = text_field.replace(r"'", " ")
        text_field = text_field.replace(r"(", " ")
        text_field = text_field.replace(r")", " ")
        text_field = text_field.replace(r"[", " ")
        text_field = text_field.replace(r"]", " ")
        text_field = text_field.replace(r"[]", " ")
        text_field = text_field.replace(r"?", " ")
        text_field = text_field.replace(r"()", " ")
        text_field = text_field.replace(r'"', " ")
        text_field = text_field.replace(r"-", " ")
        text_field = text_field.replace(r"{", " ")
        text_field = text_field.replace(r"}", " ")
        text_field = text_field.replace(r"*", " ")
        text_field = text_field.replace(r"~,!", " ")
        text_field = text_field.replace(r"@", " ")
        text_field = re.sub("[?]", " ", text_field)
        text_field = text_field.replace(r"#", " ")
        text_field = text_field.replace(r"$", " ")
        text_field = text_field.replace(r"%", " ")
        text_field = text_field.replace(r"^", " ")
        text_field = text_field.replace(r"&", " ")
        text_field = text_field.replace(r"=", " ")
        text_field = text_field.replace(r"+", " ")
        text_field = text_field.replace(r"`", " ")
        text_field = text_field.replace(r"<", " ")
        text_field = text_field.replace(r">", " ")
        text_field = text_field.replace(r"·", " ")
        text_field = re.sub("[”“]", " ", text_field)
        text_field = text_field.replace(r"//", " ")
        text_field = text_field.replace(r"|", " ")
        text_field = text_field.replace(r"|", " ")
        text_field = text_field.replace(r"&[A-Z][a-z][0-9]", " ")
        text_field = text_field.replace(r"[0-9]+", " ")
        text_field = text_field.replace(r"[a-z]+", " ")
        text_field = text_field.replace(r"[a-zA-z]", " ")
        text_field = text_field.replace(r"\[0-9a-zA-Z]", " ")
        text_field = re.sub("[–]", " ", text_field)
        text_field = text_field.replace(r"λ", " ")
        text_field = text_field.replace(r"@", "at")
        text_field = text_field.lower()
        text_field = re.sub("\s[0-9]+", " ", text_field)
        text_field = re.sub("\b[a-z]\b", " ", text_field)
        text_field = re.sub("—", " ", text_field)
        text_field = re.sub("_", " ", text_field)
        text_field = re.sub("™"," ", text_field)
        text_field = re.sub("/", " ", text_field)
        text_field = re.sub("[0-9]", " ", text_field)
        text_field = text_field.replace("nin library and unix conventions the null character is used to terminate text strings such nullterminated strings can be known in abbreviation as asciz or asciiz where here stands for zero\\nbinary oct dec hex abbreviation name\\n\\n null nul ␀ null\\n som soh ␁ start of heading\\n eoa stx ␂ start of text\\n eom etx ␃ end of text\\n eot ␄ end of transmission\\n wru enq ␅ enquiry\\n ru ack ␆ acknowledgement\\n bell bel ␇ bell\\n fe bs ␈ backspaceef\\n ht sk ht ␉ horizontal tabg\\na lf ␊ line feed\\nb vtab vt ␋ vertical tab\\nc ff ␌ form feed\\nd cr ␍ carriage returnh\\ne so ␎ shift out\\nf si ␏ shift in\\n dc dle ␐ data link escape\\n dc ␑ device control often xon\\n dc ␒ device control\\n dc ␓ device control often xoff\\n dc ␔ device control\\n err nak ␕ negative acknowledgement\\n sync syn ␖ synchronous idle\\n lem etb ␗ end of transmission block\\n can ␘ cancel\\n em ␙ end of medium\\na ss sub ␚ substitute\\nb esc ␛ ei escapej\\nc fs ␜ file separator\\nd gs ␝ group separator\\ne rs ␞ record separator\\nf us ␟ unit separator\\nf del ␡", " ")
        text_field = re.sub("[½¼¢~]", " ", text_field)
        text_field = text_field.replace('\\n', " ")
        text_field = text_field.replace("("," ")
        text_field = text_field.replace(")"," ")
        text_field = text_field.replace("#"," ")
        text_field = text_field.replace("&"," ")
        text_field = text_field.replace("\\"," ")
        text_field = ' '.join(i for i in text_field.split() if not (i.isalpha() and len(i)==1))
        return text_field

    def get_sent(text, tarr):
        with open(text, 'r') as f:
            global inp_str
            inp_str = target_list[tarr]
            text = f.read()
            text = text.replace('\\n', "")
            text = text.replace("(","")
            text = text.replace(")","")
            sentences = sent_tokenize(text)
            #for ref in range(len(top_102_words)):
            #locals()["word_sentences" + str(ref)] = [" ".join([sentences[i-1], j, sentences[i+1]]) for i,j in enumerate(sentences) if str(top_102_words[ref]) in word_tokenize(j)]
            global sent_word
            sent_word = [" ".join([sentences[i-1], j, sentences[i+1]]) for i,j in enumerate(sentences) if inp_str in word_tokenize(j)]
            #print(ref)

    def filter_sent(list_input):
        global filt_sent_
        filt_sent_= []
        for cln in list_input:
            cleaned = standardize_text(cln)
            filt_sent_.append(cleaned)
            #return filt_sent_

    target_list = []

    ## Listing all the inputs here
    number_of_target_words = int(input("Enter the number of target terms that you wish to disambiguate\n"))\


    for tar_index in range(number_of_target_words):
        tar_name = input("Enter the target term\n")
        target_list.append(tar_name)


    len(target_list)

    lower_label_int = int(input("Enter input for a starting label for which the threshold plot is to be obtained\n"))
    upper_label_int = int(input("Enter input for a ending label for which the threshold plot is to be obtained\n"))
    cluster_num_plot = int(input("Enter the cluster number for which the threshold plot is to be obtained\n"))
    thresh = float(input("Enter the threshold for the context words to be obtained\n"))

    directory_input = "elbow"

    clustering_type = "kmeans"

    text_corpus_path = str(input("Enter your path for the text corpus\n"))

    dir_path = str(input("Enter the path where you wish to save all the plots\n"))

    for tarr in range(len(target_list)):
        try:
            get_sent(text_corpus_path, tarr)
            print(len(sent_word))


            ## Taking 852 sentences due to computational limitation
            wrt = str(inp_str + 's')
            if len(sent_word) < 3000:
                print("Length Less than 3000")
                vector_bucket = []
                inter_sent = []
                sent_word_ = []
                word_bucket = []
                #di = os.mkdir("/home/amboo/Desktop/BERT/context_results/" + clustering_type + "/" + directory_input)
                #directory = os.mkdir("/home/amboo/Desktop/BERT/context_results/" + clustering_type + "/" + directory_input + "/"+ inp_str + "_" + str(1))
                dir = dir_path + clustering_type + "/" + directory_input + "/"+ inp_str + "_" + str(1)
                if os.path.exists(dir):
                    shutil.rmtree(dir)
                os.makedirs(dir)
                for gh in range(len(sent_word)):
                    sent_word_.append(sent_word[gh])
                    filter_sent(sent_word_)
                print(len(filt_sent_))
                print(len(sent_word_))
            ## Collecting the same word in different contexts
                vector_mat_list = []
                label_list = []
                for sent_par in range(len(sent_word_)):
                    try:
                        marked_text_test = "[CLS]" + " " + filt_sent_[sent_par] + " " + "[SEP]"
                        tokenized_text_test = tokenizer.tokenize(marked_text_test)
                        indexed_tokens_test = tokenizer.convert_tokens_to_ids(tokenized_text_test)
                        segment_ids_test = [1]*len(tokenized_text_test)
                        tokens_tensors_test = torch.tensor([indexed_tokens_test])
                        segments_tensors_test = torch.tensor([segment_ids_test])
                        with torch.no_grad():
                            output_test = model(tokens_tensors_test, segments_tensors_test)
                            hidden_states_test = output_test[2]
                        token_embeddings_test = torch.stack(hidden_states_test, dim=0)
                        token_embeddings_test = torch.squeeze(token_embeddings_test, dim=1)
                        token_embeddings_test = token_embeddings_test.permute(1,0,2)
                        print(token_embeddings_test.size())
                        token_vecs_sum_test = []
                        for token_test in token_embeddings_test:
                            sum_vec_test = torch.sum(token_test[-4:], dim=0)
                            token_vecs_sum_test.append(sum_vec_test)
                        token_vecs_test = hidden_states_test[-2][0]
                        sentence_embedding_test = torch.mean(token_vecs_test, dim=0)
                        print ("Our final sentence embedding vector of shape:", sentence_embedding_test.size())

                        i_test_list = []

                        for i_test, token_str_test in enumerate(tokenized_text_test):
                            if token_str_test == inp_str:
                                print(i_test, token_str_test)
                                i_test_list.append(i_test)

                        vector_mat_list.append(token_vecs_sum_test[i_test_list[0]])
                        label_str = inp_str + " " + str(sent_par)
                        label_list.append(label_str)
                        print("vector values for each instance of" + " " + inp_str)
                        print('\n')
                        print(inp_str, str(token_vecs_sum_test[i_test_list[0]][:1]))
                        print("\n" + str(sent_par))
                        inter_sent.append(sent_word_[sent_par])

                        for tok in range(len(tokenized_text_test)):
                            if tokenized_text_test[tok] != inp_str and tokenized_text_test[tok] != wrt:
                                vector_bucket.append(token_vecs_sum_test[tok])
                                word_bucket.append(tokenized_text_test[tok])

                    except (IndexError,  RuntimeError) as e:
                       # del locals()["sent_word_" + str(len(sent_word))][sent_par]
                        residual_list = []
                        final_sent_list = []
                        residual_list.append(sent_word_[sent_par])
                       # for sentr in locals()["sent_word_" + str(len(sent_word))]:
                        #    if sentr not in residual_list:
                         #       final_list.append(sentr)
                        #rint(len(final_list))
                        continue

            elif len(sent_word) > 3000:
                print("Length Greater than 3000")
                vector_bucket = []
                word_bucket = []
                inter_sent = []
                sent_word_ = []
                #di = os.mkdir("/home/amboo/Desktop/BERT/context_results/" + clustering_type + "/" + directory_input)
                #directory = os.mkdir("/home/amboo/Desktop/BERT/context_results/" + clustering_type + "/" + directory_input + "/"+ inp_str + "_" + str(1))
                #directory = os.mkdir("/home/amboo/Desktop/BERT/context_results/" + inp_str + "_" + str(1))
                dir = dir_path + clustering_type + "/" + directory_input + "/"+ inp_str + "_" + str(1)
                if os.path.exists(dir):
                    shutil.rmtree(dir)
                os.makedirs(dir)
                for gj in range(3000):
                    sent_word_.append(sent_word[gj])
                filter_sent(sent_word_)
                print(len(filt_sent_))
                print(len(sent_word_))

            ## Collecting the same word in different contexts

                vector_mat_list = []
                label_list = []
                for sent_par in range(len(sent_word_)):
                    try:
                        marked_text_test = "[CLS]" + " " + filt_sent_[sent_par] + " " + "[SEP]"
                        tokenized_text_test = tokenizer.tokenize(marked_text_test)
                        indexed_tokens_test = tokenizer.convert_tokens_to_ids(tokenized_text_test)
                        segment_ids_test = [1]*len(tokenized_text_test)
                        tokens_tensors_test = torch.tensor([indexed_tokens_test])
                        segments_tensors_test = torch.tensor([segment_ids_test])
                        with torch.no_grad():
                            output_test = model(tokens_tensors_test, segments_tensors_test)
                            hidden_states_test = output_test[2]
                        token_embeddings_test = torch.stack(hidden_states_test, dim=0)
                        token_embeddings_test = torch.squeeze(token_embeddings_test, dim=1)
                        token_embeddings_test = token_embeddings_test.permute(1,0,2)
                        print(token_embeddings_test.size())
                        token_vecs_sum_test = []
                        for token_test in token_embeddings_test:
                            sum_vec_test = torch.sum(token_test[-4:], dim=0)
                            token_vecs_sum_test.append(sum_vec_test)
                        token_vecs_test = hidden_states_test[-2][0]
                        sentence_embedding_test = torch.mean(token_vecs_test, dim=0)
                        print ("Our final sentence embedding vector of shape:", sentence_embedding_test.size())

                        i_test_list = []

                        for i_test, token_str_test in enumerate(tokenized_text_test):
                            if token_str_test == inp_str:
                                print(i_test, token_str_test)
                                i_test_list.append(i_test)

                        vector_mat_list.append(token_vecs_sum_test[i_test_list[0]])
                        label_str = inp_str + " " + str(sent_par)
                        label_list.append(label_str)
                        print("vector values for each instance of" + " " + inp_str)
                        print('\n')
                        print(inp_str, str(token_vecs_sum_test[i_test_list[0]][:1]))
                        print("\n" + str(sent_par))
                        inter_sent.append(sent_word_[sent_par])

                        for tok in range(len(tokenized_text_test)):
                            if tokenized_text_test[tok] != inp_str and tokenized_text_test[tok] != wrt:
                                vector_bucket.append(token_vecs_sum_test[tok])
                                word_bucket.append(tokenized_text_test[tok])

                    except (IndexError, RuntimeError) as e:
                        residual_list = []
                        final_list = []
                        residual_list.append(sent_word_[sent_par])
                        continue


            ## Creating the nxn matrix

            ## 3 lists have been created for the input string
            #sent_word ---- containing the sentence in which the word has been used
            # vector_mat_list ---- containing the embedded vector wrt to the sentence
            #label_list ----- labelled input string

            ## NxN Matrix Creation

            print(len(vector_mat_list))

            import numpy as np
            input_shape = len(vector_mat_list)
            target_matrix = np.zeros(shape=(input_shape,input_shape))

            matrix_list = []
            for i_ in range(input_shape):
                for j_ in range(input_shape):
                    target_matrix[i_][j_] = 1 - cosine(vector_mat_list[i_], vector_mat_list[j_])
                    matrix_list.append([i_,target_matrix[i_][j_]])
                    print(str(i_), str(j_))

            mat_dir = dir_path + "/matrices/"
            if os.path.exists(mat_dir):
                shutil.rmtree(mat_dir)
            os.makedirs(mat_dir)

            with open(mat_dir + inp_str + "_matrix.dat", "wb") as tm:
                pickle.dump(target_matrix, tm)



            target_matrix

            ## Kmeans Elbow Method

            from sklearn.cluster import KMeans
            from scipy.spatial.distance import cdist


            X = target_matrix

            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            X_ = pca.fit_transform(X)

            K = range (1,11)
            wcss = []
            distortions = []
            for i in K:
                kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state=0)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
                distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

            from kneed import KneeLocator

            kn = KneeLocator(list(K), distortions, S=1.0, curve='convex', direction='decreasing')


            import matplotlib.pyplot as plt
            figure = plt.plot(range(1,11), wcss)
            plt.title(" The Elbow Method")
            plt.xlabel("Number of Clusters")
            plt.ylabel("WCSS")
            plt.savefig(dir_path + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + "_"  + "elbow.png")
            plt.clf()

            print(wcss)

            per_dec_list = []
            k_op = 1
            K_clust = 1
            K_clustr = 1
            for wc in range(0,len(wcss)-1):
                perc_dec = (1 - (wcss[wc +1]/wcss[wc]))*100
                print(perc_dec)
                per_dec_list.append(perc_dec)
            print('\n') 

            for per in range(len(per_dec_list)):
                if per_dec_list[per] > 23:
                    k_op = wcss.index(wcss[per + 1])
                    K_clust = k_op + 1
                else:
                    K_clustr =  1

            print(K_clust)

            if K_clustr == K_clust:
                optimal_k = 1
            else:
                optimal_k = K_clust

            print(optimal_k)

            from yellowbrick.cluster import KElbowVisualizer


            X = target_matrix

            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            X_ = pca.fit_transform(X)

            kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state=0)
            visualizer = KElbowVisualizer(kmeans, k=(1,10))


            visualizer.fit(X_) 
            # Fit the data to the visualizer
            plt.xlabel("k clusters")
            plt.ylabel("distotion score")
            plt.savefig(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + "_"  + "elbow1.png")
            #visualizer.show()
            plt.clf()

            ## Intercluster Distance

            from yellowbrick.cluster import InterclusterDistance
            from sklearn.manifold import MDS

            mod = KMeans(optimal_k)
            visuals = InterclusterDistance(mod, embedding = 'tsne')

            visuals.fit(X_)        # Fit the data to the visualizer
            plt.savefig(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + "_"  + "ICD.png")
            plt.clf()

            ## Gap Statistics

            import numpy as np
            import matplotlib.pyplot as plt

            from sklearn.datasets import make_blobs
            from sklearn.metrics import pairwise_distances
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA

            #PCA Transformation
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            X_ = pca.fit_transform(X)


            def compute_inertia(a, X):
                W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
                return np.mean(W)

            def compute_gap(clustering, data, k_max=5, n_references=5):
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                reference = np.random.rand(*data.shape)
                reference_inertia = []
                for k in range(1, k_max+1):
                    local_inertia = []
                    for _ in range(n_references):
                        clustering.n_clusters = k
                        assignments = clustering.fit_predict(reference)
                        local_inertia.append(compute_inertia(assignments, reference))
                    reference_inertia.append(np.mean(local_inertia))

                ondata_inertia = []
                for k in range(1, k_max+1):
                    clustering.n_clusters = k
                    assignments = clustering.fit_predict(data)
                    ondata_inertia.append(compute_inertia(assignments, data))

                gap = np.log(reference_inertia)-np.log(ondata_inertia)
                return gap, np.log(reference_inertia), np.log(ondata_inertia)



            k_max = 5
            gap, reference_inertia, ondata_inertia = compute_gap(KMeans(), X_, k_max)


            plt.plot(range(1, k_max+1), reference_inertia,
                     '-o', label='reference')
            plt.plot(range(1, k_max+1), ondata_inertia,
                     '-o', label='data')
            plt.xlabel('k')
            plt.ylabel('log(inertia)')
            plt.savefig(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + "_"  + "gap_inertia.png")
            plt.clf()


            plt.plot(range(1, k_max+1), gap, '-o')
            plt.ylabel('gap')
            plt.xlabel('k')
            plt.savefig(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + "_"  + "gap.png")
            plt.clf()

            for n in range(len(gap)):
                if gap[n] == gap.max():
                    k_cluster_no = n +1
                    word_contexts = k_cluster_no
                    print("The predicted number of optimum clusters are {}".format(k_cluster_no))
                    print("The predicted number of contexts for the word {} are {}".format(inp_str, k_cluster_no))

            ## Silhouette Method Kmeans

            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            from sklearn.cluster import KMeans
            from sklearn import metrics
            from sklearn.metrics import silhouette_score

            sil = []
            kmax = 5

            # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
            for k in range(2, kmax+1):
                kmeans = KMeans(n_clusters = k).fit(X_)
                labels = kmeans.labels_
                sil.append(silhouette_score(X_, labels, metric = 'euclidean'))

            sil

            figure = plt.plot(range(2,kmax+1), sil)
            plt.title(" The Silhouette Method")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Sill")
            plt.savefig(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + "_"  + "sil.png")
            plt.clf()

            sil = np.array(sil)
            for g in range(len(sil)):
                if sil[g] == sil.max():
                    if sil.max() <= 0.53:
                        k_sil = 1
                        print("The predicted number of optimum clusters are {}".format(k_sil))
                        print("The predicted number of contexts for the word {} are {}".format(inp_str, k_sil))
                    else:
                        k_sil = g +2
                        print("The predicted number of optimum clusters are {}".format(k_sil))
                        print("The predicted number of contexts for the word {} are {}".format(inp_str, k_sil))

            from yellowbrick.cluster import SilhouetteVisualizer

            visualizer_sil = SilhouetteVisualizer(kmeans, colors='yellowbrick')


            visualizer_sil.fit(X_)        # Fit the data to the visualizer
            plt.savefig(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + "_"  + "sil_color.png")
            plt.clf()

            ## Cluster Analysis

            n_clusters = k_sil
            word_of_cluster_0 = []
            word_of_cluster_1 = []
            sent_of_cluster_0 = []
            sent_of_cluster_1 = []
            word_of_cluster_2 = []
            word_of_cluster_3 = []
            sent_of_cluster_2 = []
            sent_of_cluster_3 = []
            word_of_cluster_4 = []
            word_of_cluster_5 = []
            sent_of_cluster_4 = []
            sent_of_cluster_5 = []

            if n_clusters == 1:
                kmeans = KMeans(n_clusters = n_clusters , init ="k-means++",  max_iter= 300 , n_init= 10, random_state = 0 , precompute_distances = True)

                kmeans.fit(X)

                y_kmeans = kmeans.predict(X)

                for km in range(len(y_kmeans)):
                    if y_kmeans[km] == 0:
                        word_of_cluster_0.append(label_list[km])
                        sent_of_cluster_0.append(inter_sent[km])

                ##Plotting
                """User can plot the clusters if they wish"""

                #plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
                #plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
                #plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
                #plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
                #plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
                #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 150, c = 'black', label = 'Centroids')
                #plt.title('Clusters of the word {}' .format(inp_str))
                #plt.xlabel('{} Labels Along X-axis' .format(inp_str))
                #plt.ylabel('{} Labels Along Y-axis' .format(inp_str))
                #plt.legend()
                #plt.savefig("/home/amboo/Desktop/BERT/kmeans/{}.png" . format(inp_str))
                #plt.clf()
            if n_clusters == 2:
                kmeans = KMeans(n_clusters = 2 , init ="k-means++",  max_iter= 300 , n_init= 10, random_state = 0 , precompute_distances = False)

                kmeans.fit(X)

                y_kmeans = kmeans.predict(X)

                for km in range(len(y_kmeans)):
                    if y_kmeans[km] == 0:
                        word_of_cluster_0.append(label_list[km])
                        sent_of_cluster_0.append(inter_sent[km])
                    if y_kmeans[km] == 1:
                        word_of_cluster_1.append(label_list[km])
                        sent_of_cluster_1.append(inter_sent[km])

                ##Plotiing 
                """User can plot the clusters if they wish"""

                #plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
                #plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
                #plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
                #plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
                #plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
                #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 150, c = 'black', label = 'Centroids')
                #plt.title('Clusters of the word {}' .format(inp_str))
                #plt.xlabel('{} Labels Along X-axis' .format(inp_str))
                #plt.ylabel('{} Labels Along Y-axis' .format(inp_str))
                #plt.legend()
                #plt.savefig("/home/amboo/Desktop/BERT/kmeans/{}.png" . format(inp_str))
                #plt.clf()
            if n_clusters == 3:
                kmeans = KMeans(n_clusters = n_clusters , init ="k-means++",  max_iter= 300 , n_init= 10, random_state = 0 , precompute_distances = True)

                kmeans.fit(X)

                y_kmeans = kmeans.predict(X)

                for km in range(len(y_kmeans)):
                    if y_kmeans[km] == 0:
                        word_of_cluster_0.append(label_list[km])
                        sent_of_cluster_0.append(inter_sent[km])
                    if y_kmeans[km] == 1:
                        word_of_cluster_1.append(label_list[km])
                        sent_of_cluster_1.append(inter_sent[km])
                    if y_kmeans[km] == 2:
                        word_of_cluster_2.append(label_list[km])
                        sent_of_cluster_2.append(inter_sent[km])

                #Plotting
                """User can plot the clusters if they wish"""

                #plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
                #plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
                #plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
                #plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
                #plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
                #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 150, c = 'black', label = 'Centroids')
                #plt.title('Clusters of the word {}' .format(inp_str))
                #plt.xlabel('{} Labels Along X-axis' .format(inp_str))
                #plt.ylabel('{} Labels Along Y-axis' .format(inp_str))
                #plt.legend()
                #plt.savefig("/home/amboo/Desktop/BERT/kmeans/{}.png" . format(inp_str))
                #plt.clf()
            if n_clusters == 4:

                kmeans = KMeans(n_clusters = n_clusters , init ="k-means++",  max_iter= 300 , n_init= 10, random_state = 0 , precompute_distances = False)

                kmeans.fit(X)

                y_kmeans = kmeans.predict(X)

                for km in range(len(y_kmeans)):
                    if y_kmeans[km] == 0:
                        word_of_cluster_0.append(label_list[km])
                        sent_of_cluster_0.append(inter_sent[km])
                    if y_kmeans[km] == 1:
                        word_of_cluster_1.append(label_list[km])
                        sent_of_cluster_1.append(inter_sent[km])
                    if y_kmeans[km] == 2:
                        word_of_cluster_2.append(label_list[km])
                        sent_of_cluster_2.append(inter_sent[km])
                    if y_kmeans[km] == 3:
                        word_of_cluster_3.append(label_list[km])
                        sent_of_cluster_3.append(inter_sent[km])

                ##Plotting
                """User can plot the clusters if they wish"""

                #plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
                #plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
                #plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
                #plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
                #plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
                #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 150, c = 'black', label = 'Centroids')
                #plt.title('Clusters of the word {}' .format(inp_str))
                #plt.xlabel('{} Labels Along X-axis' .format(inp_str))
                #plt.ylabel('{} Labels Along Y-axis' .format(inp_str))
                #plt.legend()
                #plt.savefig("/home/amboo/Desktop/BERT/kmeans/{}.png" . format(inp_str))
                #plt.clf()
            if n_clusters == 5:
                kmeans = KMeans(n_clusters = n_clusters , init ="k-means++",  max_iter= 300 , n_init= 10, random_state = 0 , precompute_distances = True)

                kmeans.fit(X)

                y_kmeans = kmeans.predict(X)

                for km in range(len(y_kmeans)):
                    if y_kmeans[km] == 0:
                        word_of_cluster_0.append(label_list[km])
                        sent_of_cluster_0.append(inter_sent[km])
                    if y_kmeans[km] == 1:
                        word_of_cluster_1.append(label_list[km])
                        sent_of_cluster_1.append(inter_sent[km])
                    if y_kmeans[km] == 2:
                        word_of_cluster_2.append(label_list[km])
                        sent_of_cluster_2.append(inter_sent[km])
                    if y_kmeans[km] == 3:
                        word_of_cluster_3.append(label_list[km])
                        sent_of_cluster_3.append(inter_sent[km])
                    if y_kmeans[km] == 4:
                        word_of_cluster_4.append(label_list[km])
                        sent_of_cluster_4.append(inter_sent[km])

                ##Plotting
                """User can plot the clusters if they wish"""

                #plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
                #plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
                #plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
                #plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
                #plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
                #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 150, c = 'black', label = 'Centroids')
                #plt.title('Clusters of the word {}' .format(inp_str.upper()))
                #plt.xlabel('{} Labels Along Column 0' .format(inp_str.upper()))
                #plt.ylabel('{} Labels Along Column 1' .format(inp_str.upper()))
                #plt.legend()
                #plt.savefig("/home/amboo/Desktop/BERT/kmeans/{}.png" . format(inp_str))
                #plt.clf()

            ##############################################################################################################

            ### Functions


            def Sort(sub_li): 

                # reverse = None (Sorts in Ascending order) 
                # key is set to sort using second element of  
                # sublist lambda has been used 
                sub_li.sort(key = lambda x: x[1], reverse = True) 
                return sub_li 


            import pandas as pd
            def get_word_vector(word_no, word_list ,static_list ,dynamic_list, threshold, freq):
                global rel_list_
                global rel_list
                global cos_dist_
                global frame
                global sort_list
                sort_list = []
                cos_dist_ = []
                global ref
                ref = {}
                try:
                    for dynm in range(len(dynamic_list)):
                        cos_dist = 1 - cosine(static_list[word_no], dynamic_list[dynm])
                        cos_dist_.append(cos_dist)
                    ref = {"Words":word_list, "Distance": cos_dist_} 
                    frame = pd.DataFrame(ref, columns = ["Words", "Distance"])
                    rel_list = frame[frame["Distance"] >= threshold].values.tolist()
                    #rel_list.sort(reverse = False)
                    rel_list_ = Sort(rel_list)
                    for re in range(freq):
                        sort_list.append(rel_list[re])
                    #print(len(sort_list))
                    return sort_list
                except IndexError as e:
                    pass





            def get_context(cluster_no, sent_of_cluster, n_el_clus_list, threshold, frequency):
                if cluster_no == 0:
                    empty_sent = []
                    if clustering_type == 'kmeans':
                        print(word_of_cluster_0[n_el_clus_list])
                        for word in str(word_of_cluster_0[n_el_clus_list]).split():
                            if word.isdigit() == True:
                                label_number = int(word)
                        word_label = label_number
                    if clustering_type == 'birch':
                        print(word_of_cluster_b0[n_el_clus_list])
                        for word in str(word_of_cluster_b0[n_el_clus_list]).split():
                            if word.isdigit() == True:
                                label_number = int(word)
                        word_label = label_number

                    print(sent_of_cluster[n_el_clus_list])

                    #frequency = int(input("\nEnter the no of top words to be displayed\n"))
                    #frequency = 50
                    #threshold = 0.45
                    get_word_vector(word_label, word_list_refined0 ,vector_mat_list, vector_list_refined0, threshold, frequency)
                    #for (root,dirs,files) in os.walk("/home/amboo/Desktop/BERT/context_results/" + inp_str + "/"):
                    with open(dir_path + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + str(cluster_no) + "_"  + ".txt", "a") as cw:
                        cw.write("The Target Word is " + inp_str + "\n")
                        cw.write("The no of total clusters are " + str(n_clusters) + "\n")
                        cw.write("The word belongs to {} cluster".format(cluster_no) + "\n")
                        cw.write(str(word_of_cluster_0[n_el_clus_list]))
                        cw.write("\n")
                        cw.write(str(sent_of_cluster[n_el_clus_list]))
                        cw.write("\n")
                        cw.write("\n")
                        for th in sort_list:
                            cw.write(str(th))
                            cw.write("\n")
                        #cw.close()
                    cw.close()
                    return sort_list




                if cluster_no ==1:
                    if clustering_type == 'kmeans':
                        print(word_of_cluster_1[n_el_clus_list])
                        for word in str(word_of_cluster_1[n_el_clus_list]).split():
                            if word.isdigit() == True:
                                label_number = int(word)
                        word_label = label_number

                    if clustering_type == 'birch':
                        print(word_of_cluster_b1[n_el_clus_list])
                        for word in str(word_of_cluster_b1[n_el_clus_list]).split():
                            if word.isdigit() == True:
                                label_number = int(word)
                        word_label = label_number

                    print(sent_of_cluster[n_el_clus_list])

                    #frequency = int(input("\nEnter the no of top words to be displayed\n"))
                    #frequency = 50
                    #threshold = 0.46
                    get_word_vector(word_label, word_list_refined1 ,vector_mat_list, vector_list_refined1, threshold, frequency)
                    #for (root,dirs,files) in os.walk("/home/amboo/Desktop/BERT/context_results/" + inp_str + "/"):
                    with open(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + str(cluster_no) + "_"  + ".txt", "a") as cw:
                        cw.write("The Target Word is " + inp_str + "\n")
                        cw.write("The no of total clusters are " + str(n_clusters) + "\n")
                        cw.write("The word belongs to {} cluster".format(cluster_no) + "\n")
                        cw.write(str(word_of_cluster_1[n_el_clus_list]))
                        cw.write("\n")
                        cw.write(str(sent_of_cluster[n_el_clus_list]))
                        cw.write("\n")
                        cw.write("\n")
                        for th in sort_list:
                            cw.write(str(th))
                            cw.write("\n")
                    cw.close()
                    return sort_list




                if cluster_no == 2:

                    if clustering_type == 'kmeans':
                        print(word_of_cluster_2[n_el_clus_list])
                        for word in str(word_of_cluster_2[n_el_clus_list]).split():
                            if word.isdigit() == True:
                                label_number = int(word)
                        word_label = label_number

                    if clustering_type == 'birch':
                        print(word_of_cluster_b2[n_el_clus_list])
                        for word in str(word_of_cluster_b2[n_el_clus_list]).split():
                            if word.isdigit() == True:
                                label_number = int(word)
                        word_label = label_number

                    print(sent_of_cluster[n_el_clus_list])


                    #frequency = int(input("\nEnter the no of top words to be displayed\n"))
                    #frequency = 50
                    #threshold = 0.46
                    get_word_vector(word_label, word_list_refined2 ,vector_mat_list, vector_list_refined2, threshold, frequency)
                    #for (root,dirs,files) in os.walk("/home/amboo/Desktop/BERT/context_results/" + inp_str + "/"):
                    with open(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + str(cluster_no) + "_"  + ".txt", "a") as cw:
                        cw.write("The Target Word is " + inp_str + "\n")
                        cw.write("The no of total clusters are " + str(n_clusters) + "\n")
                        cw.write("The word belongs to {} cluster".format(cluster_no) + "\n")
                        cw.write(str(word_of_cluster_2[n_el_clus_list]))
                        cw.write("\n")
                        cw.write(str(sent_of_cluster[n_el_clus_list]))
                        cw.write("\n")
                        cw.write("\n")
                        for th in sort_list:
                            cw.write(str(th))
                            cw.write("\n")
                    cw.close()
                    return sort_list




                if cluster_no == 3:

                    if clustering_type == 'kmeans':
                        print(word_of_cluster_3[n_el_clus_list])
                        for word in str(word_of_cluster_3[n_el_clus_list]).split():
                            if word.isdigit() == True:
                                label_number = int(word)
                        word_label = label_number

                    if clustering_type == 'birch':
                        print(word_of_cluster_b3[n_el_clus_list])
                        for word in str(word_of_cluster_b3[n_el_clus_list]).split():
                            if word.isdigit() == True:
                                label_number = int(word)
                        word_label = label_number

                    print(sent_of_cluster[n_el_clus_list])


                    get_word_vector(word_label,word_list_refined3,vector_mat_list, vector_list_refined3, threshold, frequency)
                    with open(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + str(cluster_no) + "_"  + ".txt", "a") as cw:
                        cw.write("The Target Word is " + inp_str + "\n")
                        cw.write("The no of total clusters are " + str(n_clusters) + "\n")
                        cw.write("The word belongs to {} cluster".format(cluster_no) + "\n")
                        cw.write(str(word_of_cluster_3[n_el_clus_list]))
                        cw.write("\n")
                        cw.write(str(sent_of_cluster[n_el_clus_list]))
                        cw.write("\n")
                        cw.write("\n")
                        for th in sort_list:
                            cw.write(str(th))
                            cw.write("\n")
                    cw.close()
                    return sort_list




                if cluster_no == 4:

                    if clustering_type == 'kmeans':
                        print(word_of_cluster_4[n_el_clus_list])
                        for word in str(word_of_cluster_4[n_el_clus_list]).split():
                            if word.isdigit() == True:
                                label_number = int(word)
                        word_label = label_number

                    if clustering_type == 'birch':
                        print(word_of_cluster_b4[n_el_clus_list])
                        for word in str(word_of_cluster_b4[n_el_clus_list]).split():
                            if word.isdigit() == True:
                                label_number = int(word)
                        word_label = label_number

                    print(sent_of_cluster[n_el_clus_list])

                    #frequency = int(input("\nEnter the no of top words to be displayed\n"))
                    #frequency = 50
                    #threshold = 0.46
                    get_word_vector(word_label, word_list_refined4 ,vector_mat_list, vector_list_refined4, threshold, frequency)
                    #for (root,dirs,files) in os.walk("/home/amboo/Desktop/BERT/context_results/" + inp_str + "/"):
                    with open(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + str(cluster_no) + "_"  + ".txt", "a") as cw:
                        cw.write("The Target Word is " + inp_str + "\n")
                        cw.write("The no of total clusters are " + str(n_clusters) + "\n")
                        cw.write("The word belongs to {} cluster".format(cluster_no) + "\n")
                        cw.write(str(word_of_cluster_4[n_el_clus_list]))
                        cw.write("\n")
                        cw.write(str(sent_of_cluster[n_el_clus_list]))
                        cw.write("\n")
                        cw.write("\n")
                        for th in sort_list:
                            cw.write(str(th))
                            cw.write("\n")
                    cw.close()
                    return sort_list

            ## Approach 1 :- Obtaining Sentence wise vector embeddings of the words w.r.t clusters



            def get_embeddings(list_of_cluster_sentences, n_th_cluster):
                globals()["vector_list_" + str(n_th_cluster)] = []
                globals()["word_list_" + str(n_th_cluster)] = []

                globals()["vector_list_refined" + str(n_th_cluster)] = []
                globals()["word_list_refined" + str(n_th_cluster)] = []

                for emd in range(len(list_of_cluster_sentences)):
                    try:
                        marked_text = "[CLS]" + " " + list_of_cluster_sentences[emd] + " " + "[SEP]"
                        tokenized_text = tokenizer.tokenize(marked_text)
                        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                        segment_ids = [1]*len(tokenized_text)
                        tokens_tensors = torch.tensor([indexed_tokens])
                        segments_tensors = torch.tensor([segment_ids])
                        with torch.no_grad():
                            output = model(tokens_tensors, segments_tensors)
                            hidden_states = output[2]
                        token_embeddings_ = torch.stack(hidden_states, dim=0)
                        token_embeddings_ = torch.squeeze(token_embeddings_, dim=1)
                        token_embeddings_ = token_embeddings_.permute(1,0,2)
                        print(token_embeddings_.size())
                        token_vecs_sum_ = []
                        for token in token_embeddings_:
                            sum_vec_ = torch.sum(token[-4:], dim=0)
                            token_vecs_sum_.append(sum_vec_)
                        token_vecs = hidden_states[-2][0]
                        sentence_embedding = torch.mean(token_vecs, dim=0)
                        print ("Our final sentence embedding vector of shape:", sentence_embedding.size())
                        for tok in range(len(tokenized_text)):
                            if tokenized_text[tok] != inp_str and tokenized_text[tok] != wrt:
                                globals()["vector_list_" + str(n_th_cluster)].append(token_vecs_sum_[tok])
                                globals()["word_list_" + str(n_th_cluster)].append(tokenized_text[tok])
                    except (IndexError, RuntimeError) as e:
                        #locals()["residual_of_cluster_" + str(n_th_cluster)] = []
                        #locals()["residual_of_cluster_" + str(n_th_cluster)].append(list_of_cluster_sentences[emd])
                        continue



                for refi in range(len(globals()["word_list_" + str(n_th_cluster)])):
                    if globals()["word_list_" + str(n_th_cluster)][refi] not in globals()["word_list_refined" + str(n_th_cluster)]:
                        globals()["word_list_refined" + str(n_th_cluster)].append(globals()["word_list_" + str(n_th_cluster)][refi])
                        globals()["vector_list_refined" + str(n_th_cluster)].append(globals()["vector_list_" + str(n_th_cluster)][refi])


            if clustering_type == "kmeans":
            ## Splitting into context buckets
                if n_clusters == 1:
                    print("The number of clusters is {}".format(n_clusters))
                    get_embeddings(sent_of_cluster_0, 0)

                if n_clusters == 2:
                    print("The number of clusters is {}".format(n_clusters))
                    get_embeddings(sent_of_cluster_0, 0)
                    get_embeddings(sent_of_cluster_1, 1)

                if n_clusters == 3:
                    print("The number of clusters is {}".format(n_clusters))
                    get_embeddings(sent_of_cluster_0, 0)
                    get_embeddings(sent_of_cluster_1, 1)
                    get_embeddings(sent_of_cluster_2, 2)

                if n_clusters == 4:
                    print("The number of clusters is {}".format(n_clusters))
                    get_embeddings(sent_of_cluster_0, 0)
                    get_embeddings(sent_of_cluster_1, 1)
                    get_embeddings(sent_of_cluster_2, 2)
                    get_embeddings(sent_of_cluster_3, 3)

                if n_clusters == 5:
                    print("The number of clusters is {}".format(n_clusters))
                    get_embeddings(sent_of_cluster_0, 0)
                    get_embeddings(sent_of_cluster_1, 1)
                    get_embeddings(sent_of_cluster_2, 2)
                    get_embeddings(sent_of_cluster_3, 3)
                    get_embeddings(sent_of_cluster_4, 4)




            #print(len(word_list_refined0))
            #print(len(vector_list_refined0))
            #print(len(sentence_list_refined2))

            ## Threshold Plot

            ## Threshold Plot

            def tplot(clust, label_int):
                if clust == 0:
                    label_int_list = []
                    #locals()["word_list_refined" + str(clust)]
                    for j in range(len(word_list_refined0)):
                        label_int_list.append(j)

                    cos_d_ = []
                    label_ = label_int
                    print("\n********************************************\n")
                    #locals()["vector_list_refined" + str(clust)]
                    for dynm in range(len(vector_list_refined0)):
                        cos_d = 1 - cosine(vector_mat_list[label_],vector_list_refined0[dynm])
                        cos_d_.append(cos_d)

                    fer0 = {}
                    fer0 = {"Words" : word_list_refined0, "Word Int": label_int_list, "Distance": cos_d_}

                    frame0 = pd.DataFrame(fer0, columns = ["Words", "Word Int", "Distance"])

                    frame0.plot.scatter(x="Word Int", y = "Distance", c='DarkBlue')
                    plt.savefig(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + str(clust)+ "_" + "_" + "label" + "_" + str(label_) + "_" +"scatter.png")


                    print("The Mean Distance is {} \n********************************************\n".format(frame0["Distance"].mean()))
                    print("The Mode Distance is {} \n********************************************\n".format(frame0["Distance"].mode()))
                    print("The Median Distance is {} \n********************************************\n".format(frame0["Distance"].median()))
                    print("The Max Distance is {} \n********************************************\n".format(frame0["Distance"].max()))
                    print("\n********************************************\n")
                    print("Kurtosis Information is {} \n********************************************\n".format(frame0.kurtosis(axis=0)))
                if clust == 1:
                    label_int_list = []
                    #locals()["word_list_refined" + str(clust)]
                    for j in range(len(word_list_refined1)):
                        label_int_list.append(j)

                    cos_d_ = []
                    label_ = label_int
                    print("\n********************************************\n")
                    #locals()["vector_list_refined" + str(clust)]
                    for dynm in range(len(vector_list_refined1)):
                        cos_d = 1 - cosine(vector_mat_list[label_],vector_list_refined1[dynm])
                        cos_d_.append(cos_d)

                    fer0 = {}
                    fer0 = {"Words" : word_list_refined1, "Word Int": label_int_list, "Distance": cos_d_}

                    frame0 = pd.DataFrame(fer0, columns = ["Words", "Word Int", "Distance"])

                    frame0.plot.scatter(x="Word Int", y = "Distance", c='DarkBlue')
                    plt.savefig(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + str(clust)+ "_" + "_" + "label" + "_" + str(label_) + "_" +"scatter.png")


                    print("The Mean Distance is {} \n********************************************\n".format(frame0["Distance"].mean()))
                    print("The Mode Distance is {} \n********************************************\n".format(frame0["Distance"].mode()))
                    print("The Median Distance is {} \n********************************************\n".format(frame0["Distance"].median()))
                    print("The Max Distance is {} \n********************************************\n".format(frame0["Distance"].max()))
                    print("\n********************************************\n")
                    print("Kurtosis Information is {} \n********************************************\n".format(frame0.kurtosis(axis=0)))
                if clust == 2:
                    label_int_list = []
                    #locals()["word_list_refined" + str(clust)]
                    for j in range(len(word_list_refined2)):
                        label_int_list.append(j)

                    cos_d_ = []
                    label_ = label_int
                    print("\n********************************************\n")
                    #locals()["vector_list_refined" + str(clust)]
                    for dynm in range(len(vector_list_refined2)):
                        cos_d = 1 - cosine(vector_mat_list[label_],vector_list_refined2[dynm])
                        cos_d_.append(cos_d)

                    fer0 = {}
                    fer0 = {"Words" : word_list_refined2, "Word Int": label_int_list, "Distance": cos_d_}


                    frame0 = pd.DataFrame(fer0, columns = ["Words", "Word Int", "Distance"])

                    frame0.plot.scatter(x="Word Int", y = "Distance", c='DarkBlue')
                    plt.savefig(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + str(clust)+ "_" + "_" + "label" + "_" + str(label_) + "_" +"scatter.png")


                    print("The Mean Distance is {} \n********************************************\n".format(frame0["Distance"].mean()))
                    print("The Mode Distance is {} \n********************************************\n".format(frame0["Distance"].mode()))
                    print("The Median Distance is {} \n********************************************\n".format(frame0["Distance"].median()))
                    print("The Max Distance is {} \n********************************************\n".format(frame0["Distance"].max()))
                    print("\n********************************************\n")
                    print("Kurtosis Information is {} \n********************************************\n".format(frame0.kurtosis(axis=0)))
                if clust == 3:
                    label_int_list = []
                    for j in range(len(word_list_refined3)):
                        label_int_list.append(j)

                    cos_d_ = []
                    label_ = label_int
                    print("\n********************************************\n")
                    #locals()["vector_list_refined" + str(clust)]
                    for dynm in range(len(vector_list_refined3)):
                        cos_d = 1 - cosine(vector_mat_list[label_],vector_list_refined3[dynm])
                        cos_d_.append(cos_d)

                    fer0 = {}
                    fer0 = {"Words" : word_list_refined3, "Word Int": label_int_list, "Distance": cos_d_}
                    frame0 = pd.DataFrame(fer0, columns = ["Words", "Word Int", "Distance"])

                    frame0.plot.scatter(x="Word Int", y = "Distance", c='DarkBlue')
                    plt.savefig(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + str(clust)+ "_" + "_" + "label" + "_" + str(label_) + "_" +"scatter.png")

                    print("The Mean Distance is {} \n********************************************\n".format(frame0["Distance"].mean()))
                    print("The Mode Distance is {} \n********************************************\n".format(frame0["Distance"].mode()))
                    print("The Median Distance is {} \n********************************************\n".format(frame0["Distance"].median()))
                    print("The Max Distance is {} \n********************************************\n".format(frame0["Distance"].max()))
                    print("\n********************************************\n")
                    print("Kurtosis Information is {} \n********************************************\n".format(frame0.kurtosis(axis=0)))
                if clust == 4:
                    label_int_list = []
                    #locals()["word_list_refined" + str(clust)]
                    for j in range(len(word_list_refined4)):
                        label_int_list.append(j)

                    cos_d_ = []
                    label_ = label_int
                    print("\n********************************************\n")
                    #locals()["vector_list_refined" + str(clust)]
                    for dynm in range(len(vector_list_refined4)):
                        cos_d = 1 - cosine(vector_mat_list[label_],vector_list_refined4[dynm])
                        cos_d_.append(cos_d)

                    fer0 = {}
                    fer0 = {"Words" : word_list_refined4, "Word Int": label_int_list, "Distance": cos_d_}
                    frame0 = pd.DataFrame(fer0, columns = ["Words", "Word Int", "Distance"])

                    frame0.plot.scatter(x="Word Int", y = "Distance", c='DarkBlue')
                    plt.savefig(dir_path  + clustering_type + "/" + directory_input + "/" + inp_str + "_" + str(1) + "/" + inp_str + "_" + str(1) + "_" +"cluster" + str(clust)+ "_" + "_" + "label" + "_" + str(label_) + "_" +"scatter.png")


                    print("The Mean Distance is {} \n********************************************\n".format(frame0["Distance"].mean()))
                    print("The Mode Distance is {} \n********************************************\n".format(frame0["Distance"].mode()))
                    print("The Median Distance is {} \n********************************************\n".format(frame0["Distance"].median()))
                    print("The Max Distance is {} \n********************************************\n".format(frame0["Distance"].max()))
                    print("\n********************************************\n")
                    print("Kurtosis Information is {} \n********************************************\n".format(frame0.kurtosis(axis=0)))


            for tl in range(lower_label_int,upper_label_int):
                tplot(cluster_num_plot, tl)





            ## Collecting results for cluster 0 
            if n_clusters == 1:
                if clustering_type == 'kmeans':
                    for res0 in range(len(word_of_cluster_0)):
                        try:
                            get_context(0,sent_of_cluster_0,res0, thresh, 50)
                        except IndexError as e:
                            print("1st Exception Occured")
                            try:
                                get_context(0,sent_of_cluster_0,res0, thresh, 40)
                                continue
                            except IndexError as e1:
                                print("2nd Exception Occured")
                                try:
                                    get_context(0,sent_of_cluster_0,res0, thresh, 35)
                                    continue
                                except IndexError as e2:
                                    print("3rd Exception Occured")
                                    try:
                                        get_context(0,sent_of_cluster_0,res0, thresh, 30)
                                        continue
                                    except IndexError as e3:
                                        print("4th Exception Occured")
                                        try:
                                            get_context(0,sent_of_cluster_0,res0, thresh, 20)
                                            continue
                                        except IndexError as e4:
                                            print("5th Exception Occured")
                                            try:
                                                get_context(0,sent_of_cluster_0,res0, thresh, 10)
                                                continue
                                            except IndexError as e5:
                                                try:
                                                    print("6th Exception Occured")
                                                    get_context(0,sent_of_cluster_0,res0, thresh,5)
                                                except IndexError as e6:
                                                    print("7th Exception Occured")
                                                    get_context(0,sent_of_cluster_0,res0, thresh, 1)
                                                    continue  

            if n_clusters == 2:
                if clustering_type == 'kmeans':
                    for res0 in range(len(word_of_cluster_0)):
                        try:
                            get_context(0,sent_of_cluster_0,res0, thresh, 50)
                        except IndexError as e:
                            print("1st Exception Occured")
                            try:
                                get_context(0,sent_of_cluster_0,res0, thresh, 40)
                                continue
                            except IndexError as e1:
                                print("2nd Exception Occured")
                                try:
                                    get_context(0,sent_of_cluster_0,res0, thresh, 35)
                                    continue
                                except IndexError as e2:
                                    print("3rd Exception Occured")
                                    try:
                                        get_context(0,sent_of_cluster_0,res0, thresh, 30)
                                        continue
                                    except IndexError as e3:
                                        print("4th Exception Occured")
                                        try:
                                            get_context(0,sent_of_cluster_0,res0, thresh, 20)
                                            continue
                                        except IndexError as e4:
                                            print("5th Exception Occured")
                                            try:
                                                get_context(0,sent_of_cluster_0,res0, thresh, 10)
                                                continue
                                            except IndexError as e5:
                                                try:
                                                    print("6th Exception Occured")
                                                    get_context(0,sent_of_cluster_0,res0, thresh,5)
                                                except IndexError as e6:
                                                    print("7th Exception Occured")
                                                    get_context(0,sent_of_cluster_0,res0, thresh, 1)
                                                    continue  

                if clustering_type == 'kmeans':
                    for res1 in range(len(word_of_cluster_1)):
                        try:
                            get_context(1,sent_of_cluster_1,res1, thresh, 50)
                        except IndexError as e:
                            print("1st Exception Occured")
                            try:
                                get_context(1,sent_of_cluster_1,res0, thresh, 40)
                                continue
                            except IndexError as e1:
                                print("2nd Exception Occured")
                                try:
                                    get_context(1,sent_of_cluster_1,res1, thresh, 35)
                                    continue
                                except IndexError as e2:
                                    print("3rd Exception Occured")
                                    try:
                                        get_context(1,sent_of_cluster_1,res1, thresh, 30)
                                        continue
                                    except IndexError as e3:
                                        print("4th Exception Occured")
                                        try:
                                            get_context(1,sent_of_cluster_1,res1, thresh, 20)
                                            continue
                                        except IndexError as e4:
                                            print("5th Exception Occured")
                                            try:
                                                get_context(1,sent_of_cluster_1,res1, thresh, 10)
                                                continue
                                            except IndexError as e5:
                                                try:
                                                    print("6th Exception Occured")
                                                    get_context(1,sent_of_cluster_1,res1, thresh,5)
                                                except IndexError as e6:
                                                    print("7th Exception Occured")
                                                    get_context(1,sent_of_cluster_1,res1, thresh, 1)
                                                    continue


            if n_clusters == 3:
                if clustering_type == 'kmeans':
                    for res0 in range(len(word_of_cluster_0)):
                        try:
                            get_context(0,sent_of_cluster_0,res0, thresh, 50)
                        except IndexError as e:
                            print("1st Exception Occured")
                            try:
                                get_context(0,sent_of_cluster_0,res0, thresh, 40)
                                continue
                            except IndexError as e1:
                                print("2nd Exception Occured")
                                try:
                                    get_context(0,sent_of_cluster_0,res0, thresh, 35)
                                    continue
                                except IndexError as e2:
                                    print("3rd Exception Occured")
                                    try:
                                        get_context(0,sent_of_cluster_0,res0, thresh, 30)
                                        continue
                                    except IndexError as e3:
                                        print("4th Exception Occured")
                                        try:
                                            get_context(0,sent_of_cluster_0,res0, thresh, 20)
                                            continue
                                        except IndexError as e4:
                                            print("5th Exception Occured")
                                            try:
                                                get_context(0,sent_of_cluster_0,res0, thresh, 10)
                                                continue
                                            except IndexError as e5:
                                                try:
                                                    print("6th Exception Occured")
                                                    get_context(0,sent_of_cluster_0,res0, thresh,5)
                                                except IndexError as e6:
                                                    print("7th Exception Occured")
                                                    get_context(0,sent_of_cluster_0,res0, thresh, 1)
                                                    continue  

                if clustering_type == 'kmeans':
                    for res1 in range(len(word_of_cluster_1)):
                        try:
                            get_context(1,sent_of_cluster_1,res1, thresh, 50)
                        except IndexError as e:
                            print("1st Exception Occured")
                            try:
                                get_context(1,sent_of_cluster_1,res0, thresh, 40)
                                continue
                            except IndexError as e1:
                                print("2nd Exception Occured")
                                try:
                                    get_context(1,sent_of_cluster_1,res1, thresh, 35)
                                    continue
                                except IndexError as e2:
                                    print("3rd Exception Occured")
                                    try:
                                        get_context(1,sent_of_cluster_1,res1, thresh, 30)
                                        continue
                                    except IndexError as e3:
                                        print("4th Exception Occured")
                                        try:
                                            get_context(1,sent_of_cluster_1,res1, thresh, 20)
                                            continue
                                        except IndexError as e4:
                                            print("5th Exception Occured")
                                            try:
                                                get_context(1,sent_of_cluster_1,res1, thresh, 10)
                                                continue
                                            except IndexError as e5:
                                                try:
                                                    print("6th Exception Occured")
                                                    get_context(1,sent_of_cluster_1,res1, thresh,5)
                                                except IndexError as e6:
                                                    print("7th Exception Occured")
                                                    get_context(1,sent_of_cluster_1,res1, thresh, 1)
                                                    continue
                ## Collecting results for cluster 2
                if clustering_type == 'kmeans':
                    for res2 in range(len(word_of_cluster_2)):
                        try:
                            get_context(2,sent_of_cluster_2,res2, thresh, 50)
                        except IndexError as e:
                            print("1st Exception Occured")
                            try:
                                get_context(2,sent_of_cluster_2,res2, thresh, 40)
                                continue
                            except IndexError as e1:
                                print("2nd Exception Occured")
                                try:
                                    get_context(2,sent_of_cluster_2,res2, thresh, 35)
                                    continue
                                except IndexError as e2:
                                    print("3rd Exception Occured")
                                    try:
                                        get_context(2,sent_of_cluster_2,res2, thresh, 30)
                                        continue
                                    except IndexError as e3:
                                        print("4th Exception Occured")
                                        try:
                                            get_context(2,sent_of_cluster_2,res2, thresh, 20)
                                            continue
                                        except IndexError as e4:
                                            print("5th Exception Occured")
                                            try:
                                                get_context(2,sent_of_cluster_2,res2, thresh, 10)
                                                continue
                                            except IndexError as e5:
                                                try:
                                                    print("6th Exception Occured")
                                                    get_context(2,sent_of_cluster_2,res2, thresh,5)
                                                except IndexError as e6:
                                                    print("7th Exception Occured")
                                                    get_context(2,sent_of_cluster_2,res2, thresh, 1)
                                                    continue                            

            if n_clusters == 4:
                if clustering_type == 'kmeans':
                    for res0 in range(len(word_of_cluster_0)):
                        try:
                            get_context(0,sent_of_cluster_0,res0, thresh, 50)
                        except IndexError as e:
                            print("1st Exception Occured")
                            try:
                                get_context(0,sent_of_cluster_0,res0, thresh, 40)
                                continue
                            except IndexError as e1:
                                print("2nd Exception Occured")
                                try:
                                    get_context(0,sent_of_cluster_0,res0, thresh, 35)
                                    continue
                                except IndexError as e2:
                                    print("3rd Exception Occured")
                                    try:
                                        get_context(0,sent_of_cluster_0,res0, thresh, 30)
                                        continue
                                    except IndexError as e3:
                                        print("4th Exception Occured")
                                        try:
                                            get_context(0,sent_of_cluster_0,res0, thresh, 20)
                                            continue
                                        except IndexError as e4:
                                            print("5th Exception Occured")
                                            try:
                                                get_context(0,sent_of_cluster_0,res0, thresh, 10)
                                                continue
                                            except IndexError as e5:
                                                try:
                                                    print("6th Exception Occured")
                                                    get_context(0,sent_of_cluster_0,res0, thresh,5)
                                                except IndexError as e6:
                                                    print("7th Exception Occured")
                                                    get_context(0,sent_of_cluster_0,res0, thresh, 1)
                                                    continue  

                if clustering_type == 'kmeans':
                    for res1 in range(len(word_of_cluster_1)):
                        try:
                            get_context(1,sent_of_cluster_1,res1, thresh, 50)
                        except IndexError as e:
                            print("1st Exception Occured")
                            try:
                                get_context(1,sent_of_cluster_1,res0, thresh, 40)
                                continue
                            except IndexError as e1:
                                print("2nd Exception Occured")
                                try:
                                    get_context(1,sent_of_cluster_1,res1, thresh, 35)
                                    continue
                                except IndexError as e2:
                                    print("3rd Exception Occured")
                                    try:
                                        get_context(1,sent_of_cluster_1,res1, thresh, 30)
                                        continue
                                    except IndexError as e3:
                                        print("4th Exception Occured")
                                        try:
                                            get_context(1,sent_of_cluster_1,res1, thresh, 20)
                                            continue
                                        except IndexError as e4:
                                            print("5th Exception Occured")
                                            try:
                                                get_context(1,sent_of_cluster_1,res1, thresh, 10)
                                                continue
                                            except IndexError as e5:
                                                try:
                                                    print("6th Exception Occured")
                                                    get_context(1,sent_of_cluster_1,res1, thresh,5)
                                                except IndexError as e6:
                                                    print("7th Exception Occured")
                                                    get_context(1,sent_of_cluster_1,res1, thresh, 1)
                                                    continue
                ## Collecting results for cluster 2
                if clustering_type == 'kmeans':
                    for res2 in range(len(word_of_cluster_2)):
                        try:
                            get_context(2,sent_of_cluster_2,res2, thresh, 50)
                        except IndexError as e:
                            print("1st Exception Occured")
                            try:
                                get_context(2,sent_of_cluster_2,res2, thresh, 40)
                                continue
                            except IndexError as e1:
                                print("2nd Exception Occured")
                                try:
                                    get_context(2,sent_of_cluster_2,res2, thresh, 35)
                                    continue
                                except IndexError as e2:
                                    print("3rd Exception Occured")
                                    try:
                                        get_context(2,sent_of_cluster_2,res2, thresh, 30)
                                        continue
                                    except IndexError as e3:
                                        print("4th Exception Occured")
                                        try:
                                            get_context(2,sent_of_cluster_2,res2, thresh, 20)
                                            continue
                                        except IndexError as e4:
                                            print("5th Exception Occured")
                                            try:
                                                get_context(2,sent_of_cluster_2,res2, thresh, 10)
                                                continue
                                            except IndexError as e5:
                                                try:
                                                    print("6th Exception Occured")
                                                    get_context(2,sent_of_cluster_2,res2, thresh,5)
                                                except IndexError as e6:
                                                    print("7th Exception Occured")
                                                    get_context(2,sent_of_cluster_2,res2, thresh, 1)
                                                    continue  
                ## Collecting results for cluster 3
                if clustering_type == 'kmeans':
                    for res3 in range(len(word_of_cluster_3)):
                        try:
                            get_context(3,sent_of_cluster_3,res3, thresh, 50)
                        except IndexError as e:
                            print("1st Exception Occured")
                            try:
                                get_context(3,sent_of_cluster_3,res3, thresh, 40)
                                continue
                            except IndexError as e1:
                                print("2nd Exception Occured")
                                try:
                                    get_context(3,sent_of_cluster_3,res3, thresh, 35)
                                    continue
                                except IndexError as e2:
                                    print("3rd Exception Occured")
                                    try:
                                        get_context(3,sent_of_cluster_3,res3, thresh, 30)
                                        continue
                                    except IndexError as e3:
                                        print("4th Exception Occured")
                                        try:
                                            get_context(3,sent_of_cluster_3,res3, thresh, 20)
                                            continue
                                        except IndexError as e4:
                                            print("5th Exception Occured")
                                            try:
                                                get_context(3,sent_of_cluster_3,res3, thresh, 10)
                                                continue
                                            except IndexError as e5:
                                                try:
                                                    print("6th Exception Occured")
                                                    get_context(3,sent_of_cluster_3,res3, thresh,5)
                                                except IndexError as e6:
                                                    print("7th Exception Occured")
                                                    get_context(3,sent_of_cluster_3,res3, thresh, 1)
                                                    continue

        except ValueError as e:
            continue