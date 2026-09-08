library(dplyr)
library(stringr)
library(forcats)
library(ggplot2)
theme_update(text = element_text(size=11),
             axis.text.x = element_text(angle=90, hjust=1))
library(sciplot)
library(tidyr)
#library(hrbrthemes)
library(viridis)
library(reshape2)
library(gridExtra)
library(grid)
library(corrplot)
library(ggpubr)

#####
df_lms = read.csv('/Users/data.csv')
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

#####
df_lms_chinese = df_lms %>%
  filter(L1 == 'Chinese') %>%
  filter(level_id != 2) 

unique(df_lms_chinese$level_id)

psych::describe(df_lms_chinese)

data_long <- gather(df_lms_chinese, models, surprisals, gpt2_target_surprisal:gptneo_sent_surprisal, 
                    factor_key=TRUE)
p <- ggplot(data_long, aes(x=as.factor(level_id), y=surprisals, fill=as.factor(level_id))) +
  facet_wrap(scales = 'free', 
    ~factor(models,
                     levels = c('gpt2_target_surprisal', 
                                'distilgpt2_target_surprisal', 
                                'gptneo_target_surprisal', 
                                'gpt2_sent_surprisal', 
                                'distilgpt2_sent_surprisal', 
                                'gptneo_sent_surprisal')),
             labeller = as_labeller(c(gpt2_target_surprisal='GPT2 Target',
               distilgpt2_target_surprisal = 'DistilGPT2 Target',
               gptneo_target_surprisal = 'GPTNeo Target',
               gpt2_sent_surprisal = 'GPT2 Sentence', 
               distilgpt2_sent_surprisal = 'DistilGPT2 Sentence',
               gptneo_sent_surprisal = 'GPTNeo Sentence'))) +
  geom_boxplot(alpha=0.7) +
  stat_compare_means(comparisons = list(c("3","4"), 
                                        c("4","5")), 
                     method = 'wilcox.test', label = "p.signif") +
  
  scale_fill_manual(values=cbPalette, 
                    name="Proficiency level id"
  ) +
  ylab("Large Language Models Surprisals") +
  theme(panel.background = element_rect(fill = "white", colour = "grey50"),
        legend.text = element_text(size=17, face="bold"),
        legend.title=element_text(size=20, face="bold"),
        strip.text = element_text(size=17,face="bold", color="#993333"),
        legend.position = "bottom",
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.y = element_text(size=20,face="bold"),
        axis.text.y = element_text(face="bold", color="#993333", 
                                   size=12, angle=360))

p
ggsave('surprisalsboxplots_chinese_levels.png',p,width=12,height=10,dpi=300)

#### get effect sizes

library(rstatix)
if(require("coin")){
  ToothGrowth %>% wilcox_effsize(len ~ supp)}
head(ToothGrowth)

df_lms_chinese %>% 
  wilcox_effsize(gpt2_target_surprisal ~ level_id) 
df_lms_chinese %>%
  wilcox_effsize(distilgpt2_target_surprisal ~ level_id)
df_lms_chinese %>%
  wilcox_effsize(gptneo_target_surprisal ~ level_id)

df_lms_chinese %>%
  wilcox_effsize(gpt2_sent_surprisal ~ level_id)
df_lms_chinese %>%
  wilcox_effsize(distilgpt2_sent_surprisal ~ level_id)
df_lms_chinese %>%
  wilcox_effsize(gptneo_sent_surprisal ~ level_id)
