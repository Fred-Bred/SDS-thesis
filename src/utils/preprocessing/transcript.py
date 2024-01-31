"""
Utility functions for working with transcripts.
"""
import os
import random
import zipfile

from docx import Document
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml
import xml.etree.ElementTree as ET
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from lxml import etree

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from faker import Faker

# Function to extract text from a docx file
def get_docx_text(filename, class_df=False):
    """Extract text from a docx file"""
    document = Document(filename)

    docText = '\n\n'.join([paragraph.text for paragraph in document.paragraphs])

    if class_df:
        doc_class = filename.split('_')[-1].split('.')[0]
        return pd.DataFrame({'text': [docText], 'class': [doc_class]})
    else:
        return docText

# Function to extract comments from a docx file
def extract_comments(filename):
    """Extract all comments from a docx file"""
    doc = Document(filename)
    try:
        comments_part = doc.part.package.part_related_by('http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments')
    except KeyError:
        return []
    root = ET.fromstring(comments_part.blob)
    comments = [elem.text for elem in root.iter() if elem.tag.endswith('comment')]
    return comments

# Function to extract all the comments of document 
def get_document_comments(docxFileName):
    """Returns a dictionary with comment id as key and comment string as value"""
    ooXMLns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    comments_dict={}
    docxZip = zipfile.ZipFile(docxFileName)
    commentsXML = docxZip.read('word/comments.xml')
    et = etree.XML(commentsXML)
    comments = et.xpath('//w:comment',namespaces=ooXMLns)
    for c in comments:
        comment=c.xpath('string(.)',namespaces=ooXMLns)
        comment_id=c.xpath('@w:id',namespaces=ooXMLns)[0]
        comments_dict[comment_id]=comment
    return comments_dict

#Function to fetch all the comments in a paragraph
# def paragraph_comments(paragraph, comments_dict):
#     """Returns a list of comments in a given paragraph"""
#     ooXMLns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
#     comments=[]
#     for run in paragraph.runs:
#         comment_reference=run._r.xpath("./w:commentReference")
#         if comment_reference:
#             comment_id=comment_reference[0].xpath('@w:id',namespaces=ooXMLns)[0]
#             comment=comments_dict[comment_id]
#             comments.append(comment)
#     return comments

def paragraph_comments(paragraph, comments_dict):
    """Returns a list of comments in a given paragraph"""
    comments=[]
    for run in paragraph.runs:
        comment_reference=run._r.xpath("./w:commentReference")
        if comment_reference:
            comment_id=comment_reference[0].xpath('@w:id')[0]
            comment=comments_dict[comment_id]
            comments.append(comment)
    return comments

# Function to fetch all paragraphs with comments from a document
def comments_with_reference_paragraph(docxFileName):
    """Returns a dict with keys as paragraphs and values as comments"""
    document = Document(docxFileName)
    comments_dict=get_document_comments(docxFileName)
    comments_with_their_reference_paragraph={}
    for paragraph in document.paragraphs:  
        if comments_dict: 
            comments=paragraph_comments(paragraph,comments_dict)  
            if comments:
                comments_with_their_reference_paragraph[paragraph.text] = comments
    return comments_with_their_reference_paragraph

# Function to extract lists of text and comments from a dict where keys are paragraphs and values are comments
def extract_text_and_comments(comments_with_reference_paragraph, as_lists=False):
    """Returns a list or df of text and comments from a list of dicts where keys are paragraphs and values are comments"""
    text_list = []
    comments_list = []
    for item in comments_with_reference_paragraph:
        for key, value in item.items():
            text_list.append(key)
            comments_list.append(value)
    # text_list = [item for sublist in text_list for item in sublist]
    comments_list = [item for sublist in comments_list for item in sublist]
    # Split label strings by comma
    comments_list = [i.split(", ") for i in comments_list]
    if as_lists:
        return text_list, comments_list
    else:
        # Create df
        df = pd.DataFrame(list(zip(text_list, comments_list)), columns = ["sentences", "labels"])
        
        # Convert the list of labels into a series of strings (so that get_dummies can process it)
        df['labels'] = df['labels'].apply(lambda x: ','.join(map(str, x)))

        # Convert the labels column into dummy variables
        df = df.join(df['labels'].str.get_dummies(','))

        # Drop the labels column
        df = df.drop(columns=['labels'])

        return df

# Load data from folder
def load_data_from_folder(folder_path, as_lists=False, subscales=False):
    """Load data from folder
    args:
        folder_path (str): path to folder containing docx files
        as_lists (bool): return lists instead of df
        subscales (bool): return subscales df instead of full df
    """
    # Get all the docx files from the folder
    docx_files = [f for f in os.listdir(folder_path) if f.endswith('.docx')]

    # Extract text and comments from each docx file
    docs = [comments_with_reference_paragraph(f'{folder_path}/{f}') for f in docx_files]

    # Extract text and comments
    text_list, comments_list = extract_text_and_comments(docs, as_lists=True)

    if as_lists:
        return text_list, comments_list
    else:
        # Create df
        df = pd.DataFrame(list(zip(text_list, comments_list)), columns = ["sentence", "labels"])
        
        # Convert the list of labels into a series of strings (so that get_dummies can process it)
        df['labels'] = df['labels'].apply(lambda x: ','.join(map(str, x)))

        # Remove leading 0 from labels with regex
        df['labels'] = df['labels'].str.replace(r'\b0+', '', regex=True)

        # Convert the labels column into dummy variables
        df = df.join(df['labels'].str.get_dummies(','))

        # Drop the labels column
        df = df.drop(columns=['labels'])

        # Get a list of the column names (excluding the first one) and sort it
        columns = df.columns[1:].to_list()
        columns.sort(key=int)

        # Index the DataFrame with the first column name and the sorted list of other column names
        df = df[[df.columns[0]] + columns]

        if subscales: # create subscale df
            # Define the scale mapping
            scale_mapping = {
                1: list(range(1, 11)),
                2: list(range(11, 16)),
                3: list(range(16, 25)),
                4: list(range(25, 39)),
                5: list(range(39, 60))
            }

            # Initialize the new DataFrame with zeros
            scale_df = pd.DataFrame(data=np.zeros((len(df), len(scale_mapping))), columns=[f"scale_{i}" for i in range(1, len(scale_mapping)+1)])

            # Add the 'sentence' column from df to scale_df
            scale_df.insert(0, 'sentence', df['sentence'])

            # Iterate over the scale mapping
            for scale, scale_nums in scale_mapping.items():
                # Get the columns in df that belong to this scale
                scale_cols = [col for col in df.columns[1:] if int(col) in scale_nums]
                # If any of these columns have a value in a row, set the corresponding scale in scale_df to 1
                scale_df[f'scale_{scale}'] = df[scale_cols].any(axis=1).astype(int)
            return scale_df
        else:
            return df


# Function to create fake data
# Function to create a random Word document
def create_random_word_document(folder_path, n_docs=1):
    """Create n random Word documents"""
    fake = Faker()
    letters = ['A', 'B', 'C']

    for i in range(n_docs):
        doc = Document()
        filename = f'{folder_path}/random_doc_{i}_{random.choice(letters)}.docx'

        # Add 3-6 paragraphs
        for _ in range(random.randint(3, 6)):
            paragraph = doc.add_paragraph(fake.text())
            paragraph.add_comment(str(random.randint(0, 59)), author=fake.name(), initials='ab')

        doc.save(filename)

# Function to load patient speech turns from path or doc
def load_patient_turns(doc, prefix='P:'):
    """Load the patient speech turns from a document.
    Args:
        doc (str or Document): The document to load.
        prefix (str): The prefix used to indicate a patient speech turn."""

    # Load the document if doc is a path
    if isinstance(doc, str):
        if doc.endswith('.docx'):
            doc = Document(doc)
            # Extract the text of each paragraph
            paragraphs = [p.text for p in doc.paragraphs]
        else:
            raise ValueError("Unsupported file type. Please provide a .docx file.")

        # Filter the paragraphs to only include those that start with the specified prefix
        paragraphs = [p for p in paragraphs if p.startswith(prefix)]

        # Strip the prefix from the paragraphs
        paragraphs = [p.lstrip(prefix) for p in paragraphs]

    return paragraphs

# Function to load patient speech turns from a folder
def load_patient_turns_from_folder(folder_path, prefixes=['P:', 'PATIENT:', 'P;', 'PATIENT;']):
    """Load the patient speech turns from all documents in a folder.
    Returns a list of lists with speech turns per document.
    Args:
        folder_path (str): The path to the folder containing the documents.
        prefixes (list): The prefixes used to indicate a patient speech turn."""

    # Initialize an empty list to hold all the paragraphs
    all_paragraphs = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a Word document
        if filename.endswith('.docx'):
            # Load the document
            doc = Document(os.path.join(folder_path, filename))

            # Extract the text of each paragraph
            paragraphs = [p.text for p in doc.paragraphs]

        else:
            continue

        # Filter the paragraphs to only include those that start with any of the specified prefixes
        paragraphs = [p for p in paragraphs if any(p.startswith(prefix) for prefix in prefixes)]

        # Strip the prefix from the paragraphs
        paragraphs = [p.lstrip(prefix) for p in paragraphs for prefix in prefixes if p.startswith(prefix)]

        # Add the paragraphs to the list of all paragraphs
        all_paragraphs.append(paragraphs)

    return all_paragraphs

# Function to discard speech turns under a certain length
def filter_by_word_count(data, min_word_count=100):
    """Filter out strings that are under a specified word count.
    Args:
        data (list): A list of strings or a list of lists of strings.
        min_word_count (int): The minimum word count.
    Returns:
        A list of strings or a list of lists of strings, without strings that are under the specified word count."""

    if all(isinstance(i, str) for i in data):
        # If data is a list of strings, filter out strings that are under the specified word count
        return [s for s in data if len(s.split()) >= min_word_count]
    elif all(isinstance(i, list) for i in data):
        # If data is a list of lists of strings, filter out strings in each list that are under the specified word count
        return [[s for s in sublist if len(s.split()) >= min_word_count] for sublist in data]
    else:
        raise ValueError("Input data should be a list of strings or a list of lists of strings.")
    
# Function to split a string into chunks
def split_string(s, chunk_size):
    """Split a string into chunks of a specified word count."""
    words = s.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to split patient speech into even chunks
def split_into_chunks(data, chunk_size=100):
    """Split strings into chunks of a specified word count.
    Args:
        data (list): A list of strings or a list of lists of strings.
        chunk_size (int): The size of each chunk.
    Returns:
        A list of strings or a list of lists of strings, where the strings within each list have been combined and then split every X words."""

    if all(isinstance(i, str) for i in data):
        # If data is a list of strings, combine the strings and split into chunks
        combined_string = ' '.join(data)
        return split_string(combined_string, chunk_size)
    elif all(isinstance(i, list) for i in data):
        # If data is a list of lists of strings, combine the strings in each list and split into chunks
        return [split_string(' '.join(sublist), chunk_size) for sublist in data]
    else:
        raise ValueError("Input data should be a list of strings or a list of lists of strings.")