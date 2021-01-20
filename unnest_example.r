library(tidyverse)
library(readxl)
library(stringr)

df <- read_excel('~/Documents/eScience/projects/officeHours/data/CO_49and50_BehavioralGrading_Final.xlsx')

head(df)

# vectorised function to order and combine values
f = function(x,y) paste(sort(c(x, y)), collapse="_")
f = Vectorize(f)

dyad_df <- df %>% 
  mutate(ID1 = f(Initiator_ID, Recipient_ID),
         ID2 = as.numeric(as.factor(ID1)))

# Prepare the data for unique IDs and unnest (expand) rows 
dyad_df_complete <- dyad_df %>%
  arrange(Initiator_ID) %>%
  na.omit() # do you need to listwise delete? 

# Use unnest and do one of the ID fields at a time.  
unnest_df <- dyad_df_complete %>%
  mutate(Recipient_ID = strsplit(as.character(Recipient_ID), ',')) %>%
  unnest(Recipient_ID)  %>%
  mutate(Initiator_ID = strsplit(as.character(Initiator_ID), ',')) %>%
  unnest(Initiator_ID) %>%
  mutate(Recipient_ID = str_trim(Recipient_ID),
         Initiator_ID = str_trim(Initiator_ID)) %>%
  filter(str_detect(Recipient_ID, '^\\d+$'))
