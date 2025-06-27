import spacy
import stanza
import pandas as pd


def extraer_relaciones_spacy(textos, debug=True, implicito=False):
    """
    Extrae relaciones sujeto-verbo-objeto usando spaCy

    Args:
        textos (list): Lista de textos a procesar
        debug (bool): Si mostrar información de debug
        implicito (bool): Si buscar sujetos implicitos o no

    Returns:
        list: Lista de diccionarios con las relaciones encontradas
    """
    nlp = spacy.load("es_core_news_lg")
    relaciones = []
    entidadesCompleta = []

    for texto in textos:
        doc = nlp(texto)
        oraciones = list(doc.sents)
        entidades = list(doc.ents)
        entidadesCompleta += entidades

        for oracion in oraciones:
            if debug: print(f"\nOración: {oracion.text}")
            verbos = [token for token in oracion if token.pos_ == "VERB"]

            for verbo in verbos:
                if debug: print(f"Verbo encontrado: {verbo.text} ({verbo.lemma_})")
                sujetos = set()
                objetos = set()
                complemento_verbo = ""


                for token in oracion:
                    # Primero se busca sujetos explícitos
                    if token.dep_ in ["nsubj", "nsubj:pass"] and token.head == verbo:
                        if token.ent_type_:
                            for ent in entidades:
                                if token in ent and token.ent_type_ in ["PER", "ORG"]:
                                    sujetos.add(ent.text)
                                    if debug: print(f"Sujeto encontrado: {ent.text}")
                    # Después se busca un complemento directo o indirecto como objeto
                    if token.dep_ in ["obj", "iobj", "obl"] and verbo == token.head:
                        if token.pos_ != "PRON":
                            if token.ent_type_:
                                for ent in entidades:
                                    if token in ent:
                                        objetos.add(ent.text)
                                        if debug: print(f"Objeto Entidad encontrado: {ent.text}")
                            else:
                                for chunk in oracion.noun_chunks:
                                    if token in chunk:
                                        objetos.add(chunk.text)
                                        if debug: print(f"Objeto Chunk encontrado: {chunk.text}")
                    # Para complementar al verbo
                    if token.dep_ in ["compound", "case", "xcomp"] and token.i == (verbo.i + 1):
                        if debug: print(f"Complemento del verbo encontrado: {token.text} ({token.dep_})")
                        complemento_verbo = token.text


                # Si no encontramos sujeto explícito, buscamos en tokens anteriores
                if implicito:
                    if not sujetos:
                        for token_anterior in reversed(list(oracion)):
                            if sujetos: break
                            if token_anterior.i < verbo.i and token_anterior.ent_type_ in ["PER", "ORG"]:
                                for ent in entidades:
                                    if token_anterior in ent:
                                        sujetos.add(ent.text)
                                        if debug: print(f"Sujeto implicito encontrado: {ent.text}")
                                        break

                verbo_final = ""
                if complemento_verbo:
                    verbo_final = f"{verbo.lemma_} {complemento_verbo}"
                else:
                    verbo_final = verbo.lemma_


                # Generar combinaciones únicas de sujeto-verbo-objeto
                for sujeto in sujetos:
                    for objeto in objetos:
                        relacion = {
                            "sujeto": sujeto,
                            "verbo": verbo_final,
                            "objeto": objeto,
                            "contexto": oracion.text
                        }
                        relaciones.append(relacion)
                        if debug:
                            print(f"- {sujeto} → {verbo.lemma_} → {objeto}")

    return relaciones


def extraer_relaciones_stanza(textos, debug=True, implicito=False):
    """
    Extrae relaciones sujeto-verbo-objeto usando Stanza

    Args:
        textos (list): Lista de textos a procesar
        debug (bool): Si mostrar información de debug
        implicito (bool): Si buscar sujetos implicitos o no

    Returns:
        list: Lista de diccionarios con las relaciones encontradas
    """
    nlp = stanza.Pipeline('es', processors='tokenize,mwt,pos,lemma,depparse,ner')
    relaciones = []
    entidadesCompleta = []

    for texto in textos:
        doc = nlp(texto)

        for oracion in doc.sentences:
            entidades = list(oracion.ents)
            entidadesCompleta += entidades

            if debug: print(f"\nOración: {oracion.text}")
            verbos = [token for token in oracion.words if token.upos == "VERB"]

            for verbo in verbos:
                if debug: print(f"Verbo encontrado: {verbo.text} ({verbo.lemma})")
                sujetos = set()
                objetos = set()
                complemento_verbo = ""


                for token in oracion.words:
                    # Buscar sujetos explícitos relacionados con el verbo
                    if token.deprel in ["nsubj", "nsubj:pass"] and token.head == verbo.id:
                        for ent in entidades:
                            if token in ent.words and ent.type in ["PER", "ORG"]:
                                sujetos.add(ent.text)
                                if debug: print(f"Sujeto encontrado: {ent.text}")
                    # Buscar objetos relacionados con el verbo
                    if token.deprel in ["obj", "iobj", "obl"] and token.head == verbo.id and token.upos != "PRON":
                        encontrado = False
                        for ent in entidades:
                            if token in ent.words:
                                objetos.add(ent.text)
                                if debug: print(f"Objeto Entidad encontrado: {ent.text}")
                                encontrado = True
                                break

                        if not encontrado:
                            # Construir un 'chunk' manualmente a partir del token
                            chunk = [token]
                            for other in oracion.words:
                                if other.head == token.id and other.deprel in ["det", "amod", "nmod", "compound",
                                                                               "nummod", "acl"]:
                                    chunk.append(other)
                                elif token.head == other.id and token.deprel in ["det", "amod", "compound", "nummod"]:
                                    chunk.append(other)

                            chunk = list(set(chunk))
                            chunk.sort(key=lambda x: int(x.id))
                            objeto_texto = " ".join(t.text for t in chunk)
                            objetos.add(objeto_texto)
                            if debug: print(f"Objeto Chunk encontrado: {objeto_texto}")
                    if token.deprel in ["compound", "case", "xcomp", "advmod"] and token.id == verbo.id + 1:
                            if debug: print(f"Complemento del verbo encontrado: {token.text} ({token.deprel})")
                            complemento_verbo = token.text

                # Si no encontramos sujeto explícito, buscamos en tokens anteriores
                if implicito:
                    if not sujetos:
                        for token_anterior in reversed(list(oracion.words)):
                            if sujetos: break
                            if int(token_anterior.id) < int(verbo.id):
                                for ent in entidades:
                                    if token_anterior in ent.words and ent.type in ["PER", "ORG"]:
                                        sujetos.add(ent.text)
                                        if debug: print(f"Sujeto implicito encontrado: {sujeto}")
                                        break

                verbo_final = ""
                if complemento_verbo:
                    verbo_final = f"{verbo.lemma} {complemento_verbo}"
                else:
                    verbo_final = verbo.lemma


                # Crear combinaciones sujeto-verbo-objeto
                for sujeto in sujetos:
                    for objeto in objetos:
                        relaciones.append({
                            "sujeto": sujeto,
                            "verbo": verbo_final,
                            "objeto": objeto,
                            "contexto": oracion.text
                        })
                        if debug: print(f"- {sujeto} → {verbo.lemma} → {objeto}")

    return relaciones


def extraer_relaciones(textos, libreria="spacy", debug=True, implicito=False):
    """
    Función principal para extraer relaciones usando spaCy o Stanza

    Args:
        textos (list): Lista de textos a procesar
        libreria (str): "spacy" o "stanza"
        debug (bool): Si mostrar información de debug

    Returns:
        list: Lista de diccionarios con las relaciones encontradas
    """
    if libreria.lower() == "spacy":
        return extraer_relaciones_spacy(textos, debug, implicito)
    elif libreria.lower() == "stanza":
        return extraer_relaciones_stanza(textos, debug, implicito)
    else:
        raise ValueError("La librería debe ser 'spacy' o 'stanza'")


def mostrar_relaciones(relaciones, contexto=False):
    """
    Muestra las relaciones encontradas por consola

    Args:
        relaciones (list): Lista de relaciones
        contexto (bool): Si mostrar el contexto de cada relación
    """
    print("\nRelaciones (Sujeto → Verbo → Objeto):\n")
    for r in relaciones:
        if contexto:
            print(f"- {r['sujeto']} → {r['verbo']} → {r['objeto']} ({r['contexto']})")
        else:
            print(f"- {r['sujeto']} → {r['verbo']} → {r['objeto']}")


def relaciones_a_csv(relaciones, nombre_archivo="relaciones.csv"):
    """
    Convierte las relaciones a un archivo CSV

    Args:
        relaciones (list): Lista de relaciones
        nombre_archivo (str): Nombre del archivo CSV a generar

    Returns:
        pandas.DataFrame: DataFrame con las relaciones
    """
    df = pd.DataFrame(relaciones)
    df = df.rename(columns={
        'sujeto': 'Head',
        'verbo': 'Relation',
        'objeto': 'Tail',
        'contexto': 'Context'
    })

    df.to_csv(nombre_archivo, index=False, encoding='utf-8')
    print(f"Archivo CSV guardado como: {nombre_archivo}")

    return df


# Ejemplo de uso:
if __name__ == "__main__":
    # Ejemplo de uso del script
    textos_ejemplo = [
        "Juan trabaja en la empresa Microsoft.",
        "María dirige el departamento de ventas."
    ]

    # Extraer relaciones con spaCy
    relaciones = extraer_relaciones(textos_ejemplo, libreria="spacy", debug=True)

    # Mostrar resultados
    mostrar_relaciones(relaciones)

    # Convertir a CSV
    df = relaciones_a_csv(relaciones, "ejemplo_relaciones.csv")

    print(f"\nSe encontraron {len(relaciones)} relaciones.")