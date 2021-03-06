from peewee import *
import Database.db as db
import config


def get_articles():
  if config.DEBUG is True:
    return db.Article.select(db.Article.title, db.Article.abstract).limit(10).execute()
  else:
    query = db.Article.select(db.Article.title, db.Article.abstract)

    if config.EXCLUDE_REVIEWS is True:
      query = query.where(db.Article.review_id.not_in(config.REMOVED_REVIEWS))

    return query.execute()


def get_pmids():
  if config.DEBUG is True:
    pmids = db.Article.select(db.Article.pubmed_id).limit(10).execute()
  else:
    query = db.Article.select(db.Article.pubmed_id)

    if config.EXCLUDE_REVIEWS is True:
      query = query.where(db.Article.review_id.not_in(config.REMOVED_REVIEWS))

    pmids = query.execute()

  return [x.pubmed_id for x in pmids]


def get_labels():
  if config.DEBUG is True:
    labels = db.Article.select(db.Article.included).limit(10).execute()
  else:
    query = db.Article.select(db.Article.included)

    if config.EXCLUDE_REVIEWS is True:
      query = query.where(db.Article.review_id.not_in(config.REMOVED_REVIEWS))

    labels = query.execute()

  return [x.included for x in labels]


def get_review_names():
  if config.DEBUG is True:
    review_id_query = db.Article.select(db.Article.review_id).limit(10).execute()
  else:
    query = db.Article.select(db.Article.review_id)

    if config.EXCLUDE_REVIEWS is True:
      query = query.where(db.Article.review_id.not_in(config.REMOVED_REVIEWS))

    review_id_query = query.execute()

  review_ids = [x.review_id for x in review_id_query]

  return review_ids


def get_review_indices():
  query = db.Article.select(db.Article.review_id)

  if config.EXCLUDE_REVIEWS is True:
    query = query.where(db.Article.review_id.not_in(config.REMOVED_REVIEWS))

  review_id_query = query.execute()

  review_ids = [x.review_id for x in review_id_query]

  indices = []

  last_review = None
  first_spotted = 0

  for i in range(len(review_ids)):
    review_id = review_ids[i]

    if last_review is None:
      last_review = review_id

    if i < len(review_ids) - 1:
      next_review_id = review_ids[i]
      
      if next_review_id != last_review:
        indices.append((first_spotted, i))

        first_spotted = i
        last_review = next_review_id
    else:
      indices.append((first_spotted, i + 1))

  return indices