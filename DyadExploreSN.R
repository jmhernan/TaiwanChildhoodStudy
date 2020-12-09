library(tidyverse)
library(readxl)
library(stringr)
library(igraph) 
library(intergraph)
library(ggrepel)
library(ggnetwork)
library(pander) 

df <- read_excel('~/Documents/eScience/projects/officeHours/data/CO_49and50_BehavioralGrading_Final.xlsx')

head(df)

df %>%
  mutate(ID = group_indices(df, .dots = c('Initiator_ID', 'Recipient_ID')))

# vectorised function to order and combine values
f = function(x,y) paste(sort(c(x, y)), collapse="_")
f = Vectorize(f)

dyad_df <- df %>% 
  mutate(ID1 = f(Initiator_ID, Recipient_ID),
         ID2 = as.numeric(as.factor(ID1)))

# Looking at dyads
dyad_df %>%
  group_by(ID2, Behavior) %>%
  summarize(n = n()) %>%
  filter(ID2 == 54)

dyad_df %>%
  filter(ID2 == 54)

dyad_df %>%
  filter(Initiator_ID == 49)

# Social Network Approach 

# recipients 
dyad_df %>%
  group_by(Recipient_ID, Behavior) %>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>% 
  top_n(10, n)

# initiators
dyad_df %>%
  group_by(Initiator_ID, Behavior) %>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>% 
  top_n(10, n)

## Prepare the data for unique IDs and unnest (expand) rows 
network_df <- dyad_df %>%
  select(Initiator_ID, Recipient_ID) %>%
  arrange(Initiator_ID) %>%
  na.omit() %>%
  distinct() 

network_unnest1 <- network_df %>%
  mutate(Recipient_ID = strsplit(as.character(Recipient_ID), ',')) %>%
  unnest(Recipient_ID)

network_unnest2 <- network_unnest1 %>%
  mutate(Initiator_ID = strsplit(as.character(Initiator_ID), ',')) %>%
  unnest(Initiator_ID) %>%
  mutate(Recipient_ID = str_trim(Recipient_ID),
         Initiator_ID = str_trim(Initiator_ID)) %>%
  filter(str_detect(Recipient_ID, '^\\d+$'))

# WE DO NOT NEED THIS TYPE OF CLUSTER ANALYSIS FOR THESE SUBSETS
clusters <- clusters(graph.data.frame(network_unnest2))

with(clusters,
       data.frame(
         ID = names(membership),
         group = membership,
         group_size = csize[membership]
       )
  ) %>%
  arrange(group) 

# NETWORK GRAPH AND COMPONENTS FOR FUTURE 
# nw test 
network <- dyad_df %>%
  select(Initiator_ID, Recipient_ID) %>%
  xtabs(~ Initiator_ID + Recipient_ID, data = .) 

initiator <- unique(dyad_df$Initiator_ID)


recipients <- intersect(initiator, colnames(network))
unmentioned_initiators <- setdiff(rownames(network), recipients)
missing_initiators <- matrix(0, ncol = length(unmentioned_initiators),
                             nrow = nrow(network), dimnames = list(rownames(network), unmentioned_initiators))

graph_data <- network[, recipients]
graph_data <- cbind(graph_data, missing_initiators)

remove = setdiff(colnames(graph_data), rownames(graph_data))

graph_data <- graph_data[, !(colnames(graph_data) %in% remove)]

graph <- graph_from_adjacency_matrix(graph_data, 
                                     mode = "directed", 
                                     weighted = T, 
                                     add.colnames = T, 
                                     add.rownames = T)

components(graph)
plot(graph)
maximal.cliques(graph)
