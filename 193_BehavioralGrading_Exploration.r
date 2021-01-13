## # Create Dyad IDs for Exploratory Analysis

library(tidyverse)
library(readxl)

df <- read_excel('/Users/jing/Documents/MyDocuments/CurrentProjects/WolfProject/DataOrganizationAnalyses/ChildObservation/BehavioralGrading/193/BG/193_BehavioralGrading_All.xlsx')

head(df)

df %>%
    mutate(ID = group_indices(df, .dots = c('Initiator_ID', 'Recipient_ID')))

install.packages("xlsx")
library("xlsx")

getwd()

setwd("/Users/jing/Documents/MyDocuments/CurrentProjects/WolfProject/DataOrganizationAnalyses/ChildObservation/BehavioralGrading/193/BG/")

getwd()

df_dyad_ID <- df %>%
    mutate(ID = group_indices(df, .dots = c('Initiator_ID', 'Recipient_ID')))
write.xlsx(df_dyad_ID, file = "ProgrammaticAnalyses/193_BG_all.xlsx",
      sheetName = "dyad_ID", append = FALSE)

# vectorised function to order and combine values
f = function(x,y) paste(sort(c(x, y)), collapse="_")
f = Vectorize(f)

dyad_df <- df %>% 
  mutate(ID1 = f(Initiator_ID, Recipient_ID),
         ID2 = as.numeric(as.factor(ID1)))

write.xlsx(dyad_df, file = "ProgrammaticAnalyses/193_BG_all.xlsx",
      sheetName = "ID1ID2", append = TRUE)

## for now just look at dyadic relationships

dyad_df %>%
    group_by(ID2, Behavior) %>%
    summarize(n = n()) %>%
    filter(ID2 == 51)

dyad_df %>%
    filter(Initiator_ID == 193)

dyad_df %>%
    filter(ID2 == 51)

dyad_df %>%
    filter(Recipient_ID == 193)

dyad_df %>%
    filter(Recipient_ID == 193 | Initiator_ID == 193)

dyad_df_new <- dyad_df %>%
    filter(Recipient_ID == 193 | Initiator_ID == 193)
write.xlsx(dyad_df_new, file = "ProgrammaticAnalyses/193_dyads.xlsx",
      sheetName = "All", append = FALSE)


dyad_df_new %>%
    group_by(ID1, Behavior) %>%
    summarize(n = n())

df_193_dyads <- dyad_df_new %>%
    group_by(ID1, Behavior) %>%
    summarize(n = n())

df_193_recipient <- dyad_df %>%
    filter(Recipient_ID == 193)
write.xlsx(df_193_recipient, file = "ProgrammaticAnalyses/193_dyads.xlsx",
      sheetName = "recipient", append = TRUE)

df_193_initiator <- dyad_df %>%
    filter(Initiator_ID == 193)
write.xlsx(df_193_initiator, file = "ProgrammaticAnalyses/193_dyads.xlsx",
      sheetName = "initiator", append = TRUE)

df_193_summary <- dyad_df_new %>%
    group_by(ID1, Behavior) %>%
    summarize(n = n())

getwd()



## I imported the data. But this data is the same as dyad_df_new, could have skipped a few steps.
dyads_193 <- read_excel("ProgrammaticAnalyses/193_dyads.xlsx", sheet = "All")

head(dyads_193)

## change Score from numerical to factor variable

str(dyads_193$Score)

dyads_193$Score <- factor(dyads_193$Score)

str(dyads_193$Score)

# basic counts

unique(dyads_193$Behavior)
summary(dyads_193$Behavior)
n_distinct(dyads_193$Behavior)
unique(dyads_193$Obs_Index)
summary(dyads_193$Obs_Index)
n_distinct(dyads_193$Obs_Index)
head(dyads_193)

dyads_193 %>%
  select(Behavior) %>%
    group_by(Behavior) %>%
   summarise(n=n())

## repeat this to df_193_recipient
str(df_193_recipient$Score)

df_193_recipient$Score <- factor(df_193_recipient$Score)

str(df_193_recipient$Score)

df_193_recipient %>%
    select(Behavior) %>%
    n_distinct()

df_193_recipient %>%
    select(Obs_Index) %>%
    n_distinct()

df_193_recipient %>%
    select(Behavior) %>%
    group_by(Behavior) %>%
    summarise(n=n())

## repeat this to df_193_initiator
str(df_193_initiator$Score)
df_193_initiator$Score <- factor(df_193_initiator$Score)
str(df_193_initiator$Score)

df_193_initiator %>%
    select(Behavior) %>%
    n_distinct()

df_193_initiator %>%
    select(Obs_Index) %>%
    n_distinct()

df_193_initiator %>%
    select(Behavior) %>%
    group_by(Behavior) %>%
    summarise(n=n())

## rank who interacted with 193
df_193_initiator %>%
    group_by(Recipient_ID) %>%
    summarize(n=n()) %>%
    arrange(desc(n)) %>%
    top_n(10,n)

df_193_recipient %>%
    group_by(Initiator_ID) %>%
    summarize(n=n()) %>%
    arrange(desc(n)) %>%
    top_n(10,n)

dyad_df %>%
    group_by(Initiator_ID) %>%
    summarize(n=n()) %>%
    arrange(desc(n)) %>%
    top_n(10,n)

dyad_df %>%
    group_by(Recipient_ID) %>%
    summarize(n=n()) %>%
    arrange(desc(n)) %>%
    top_n(10,n)

df_193_recipient_new <- df_193_recipient %>%
  select(Behavior, Score)
head(df_193_recipient_new)

df_193_recipient_pivot <- gather (df_193_recipient_new, key, value, -Score) %>%
 count(Score, key, value) %>%
 spread(value, n, fill =0)
head(df_193_recipient_pivot)
write.xlsx(df_193_recipient_pivot, file = "ProgrammaticAnalyses/193_dyads.xlsx",
      sheetName = "193_recipient_pivot", append = TRUE)

df_193_initiator_new <- df_193_initiator %>%
  select(Behavior, Score)
head(df_193_initiator_new)

df_193_initiator_pivot <- gather (df_193_initiator_new, key, value, -Score) %>%
 count(Score, key, value) %>%
 spread(value, n, fill =0)
head(df_193_initiator_pivot)
write.xlsx(df_193_initiator_pivot, file = "ProgrammaticAnalyses/193_dyads.xlsx",
      sheetName = "193_initiator_pivot", append = TRUE)


