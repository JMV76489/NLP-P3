import csv #Importar librerias de CSV
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer #Importar Vectorizador de conteo y tfidf
import pickle #Importar libreria de pickle
import os #Importar libreria de sistema operativo

NORMALIZED_DATA_FILENAME = "normalized_data.csv" #Nombre del archivo de datos normalizados
TOKEN_PATTERN = r'(?u)\w+|\w+\n|\.|\¿|\?|;|:|' #Regex del vectorizador
DUMP_FILENAME = "vectors_dump.pkl"

#Función para dumpear una lista de vectores
def pickle_dump_vector_list(dump_filename : str, list : list):
    dump_file = open(dump_filename,"wb") #Abrir archivo de dumpeo
    pickle.dump(list,dump_file) #Dumpear vector al archivo
    dump_file.close() #Cerrar archivo de dumpeo


#Función para vectorizar y dumpear vector
# vector_representaion_type -> 0 = frequency ;  1 = binarized ; 2 = TF-IDF
def vectorize(corpus : list, vectorizer_token_pattern : str, vector_representation_type : int, vector_n_gram_range : tuple):
    # #Verificar si el archivo de dumpeo existe
    # if(os.path.exists(dump_filename)):
    #     print("Archivo \"", dump_filename,"\" ya existe")
    # #En caso de que el archivo de dumpeo no existe se creara el vector y se dumpeara en el archivo especificado por "dump_filename"    
    # else:
        #Verificar que el tipo de representación sea de tipo 2
        x = None
        if(vector_representation_type == 2):

            #Crear vectorizador de tfidf
            count_vectorizer = TfidfVectorizer(ngram_range=vector_n_gram_range,token_pattern=vectorizer_token_pattern)
            x = count_vectorizer.fit_transform(corpus) #Realizar fit y transform al vectorizador de tf-idf
            print(x.toarray())
            return x

        #Si la representación no es de tipo TF-IDF entonces es de frequencia o binarizado    
        else:
            is_binary = None
            #Verificar que el tipo de representación sea de tipo 0
            if(vector_representation_type == 0):
                is_binary = False
            #Verificar que el tipo de representación sea de tipo 1    
            elif(vector_representation_type == 1):
                is_binary = True  
            #En caso de que la representación sea una que no haya sido establecida    
            else:
                print("Tipo de representación desconocido")
                return x;    

            #Crear vectorizador de conteo
            count_vectorizer = CountVectorizer(binary=is_binary,ngram_range=vector_n_gram_range,token_pattern=vectorizer_token_pattern)
            x = count_vectorizer.fit_transform(corpus) #Realizar fit y transform al vectorizador de conteo
            print(x.toarray())
            return x

#Abrir archivo con datos normalizados
with(open(NORMALIZED_DATA_FILENAME,encoding="UTF-8")) as csv_data:

    csv_reader = csv.reader(csv_data) #Lector de archivos csv
    corpus_title = [] #Lista que es el corpus de los titulos
    corpus_desc = [] #Lista que es el corpus de las descripciones
    corpus_titledesc = [] #Lista que es el corupos de las descripciones concatenadas con los titulos
    corpus_list = [corpus_title,corpus_desc,corpus_titledesc] #Lista de corpus
    ngram_list = [(1,1),(2,2)]
    next(csv_reader) #Saltarse las cabeceras

    #Leer el csv fila por fila para realizar el guaradado de corpus
    for row in csv_reader:
        corpus_title.append(row[1]) #Agregar al corpus de titulos el titulo de la fila actual
        corpus_desc.append(row[2]) #Agregar al corpus de descripciones la descripcion de la fila actual
        corpus_titledesc.append(row[1] + row[2]) #Agregar al corpus de titulos y descripciones concatenados; la descripcion y el titulo concatenados de la fila

    #Verificar si el archivo de dumpeo existo
    if(os.path.exists(DUMP_FILENAME)):
        print("Cargando lista de vectores del archivo")
        #Si si existe, entonces lo abrira para cargarlo
        with open(DUMP_FILENAME,"rb") as dump_file:
            vector_list = pickle.load(dump_file) #Cargar lista de vectores
            #Debug: Imprimir elementos de la lista de vectores
            for v in vector_list:
                print(v.toarray())
    else:
        vectors_list = [] #Lista de vectores donde se almacenaran los vectores
        #Recorrer en lista de corpus
        for corpus in corpus_list:
            #Recorrer en lista de ngramas
            for ngram in ngram_list:
                #Recorrer en los tipos de representación del vector
                for type in range(0,3):
                    vectors_list.append(vectorize(corpus,TOKEN_PATTERN,type,ngram)) #Agregar representación vectorial del corpus a la lista de vectores
        pickle_dump_vector_list(DUMP_FILENAME,vectors_list) #Dumpear lista de vectores