library(catboost)
library(tidyverse)
library(janitor)
theme_set(theme_light())

# Data

zv <-
  c(
    "auth_3mth_post_acute_dia",
    "bh_ip_snf_net_paid_pmpm_cost_9to12m_b4",
    "auth_3mth_acute_ckd",
    "bh_ip_snf_net_paid_pmpm_cost_3to6m_b4",
    "auth_3mth_post_acute_trm",
    "auth_3mth_post_acute_rsk",
    "auth_3mth_acute_vco",
    "auth_3mth_dc_ltac",
    "auth_3mth_post_acute_inj",
    "bh_ip_snf_mbr_resp_pmpm_cost_6to9m_b4",
    "auth_3mth_post_acute_ben",
    "auth_3mth_acute_ccs_048",
    "bh_ip_snf_net_paid_pmpm_cost_0to3m_b4",
    "auth_3mth_hospice",
    "auth_3mth_acute_bld",
    "auth_3mth_acute_ccs_030",
    "auth_3mth_acute_neo",
    "auth_3mth_post_acute_vco",
    "auth_3mth_post_acute_dig",
    "auth_3mth_post_acute_hdz",
    "bh_ip_snf_mbr_resp_pmpm_cost_3to6m_b4",
    "auth_3mth_acute_ccs_172",
    "auth_3mth_acute_ccs_154",
    "bh_ip_snf_mbr_resp_pmpm_cost_9to12m_b4",
    "auth_3mth_post_acute_cir",
    "auth_3mth_post_acute_cer",
    "auth_3mth_post_acute_mus",
    "bh_ip_snf_net_paid_pmpm_cost_6to9m_b4",
    "auth_3mth_post_acute_sns",
    "auth_3mth_acute_can",
    "auth_3mth_post_acute_men",
    "auth_3mth_acute_ccs_153",
    "auth_3mth_transplant",
    "auth_3mth_acute_ccs_227",
    "auth_3mth_ltac",
    "auth_3mth_acute_men",
    "auth_3mth_acute_ccs_086",
    "auth_3mth_acute_cer",
    "auth_3mth_acute_trm",
    "auth_3mth_acute_dia",
    "auth_3mth_snf_direct",
    "auth_3mth_acute_ccs_067",
    "auth_3mth_acute_ccs_043",
    "auth_3mth_acute_ner",
    "auth_3mth_acute_ccs_094",
    "auth_3mth_post_acute_cad",
    "auth_3mth_acute_ccs_044",
    "auth_3mth_post_acute_ckd",
    "auth_3mth_post_acute_ner",
    "auth_3mth_post_acute_chf",
    "auth_3mth_acute_ccs_042",
    "auth_3mth_post_acute_inf",
    "auth_3mth_post_acute_gus",
    "auth_3mth_post_acute_end"
  )

conv <-
  c(
    "auth_3mth_dc_home",
    "bh_ncdm_ind",
    "atlas_retirement_destination_2015_upda",
    "auth_3mth_dc_no_ref",
    "auth_3mth_dc_snf",
    "auth_3mth_acute_end",
    "auth_3mth_psychic",
    "atlas_hiamenity",
    "auth_3mth_bh_acute",
    "auth_3mth_acute_chf",
    "auth_3mth_dc_hospice",
    "auth_3mth_acute_skn",
    "atlas_hipov_1115",
    "auth_3mth_acute_res",
    "atlas_foodhub16",
    "auth_3mth_acute_dig",
    "auth_3mth_dc_acute_rehab",
    "atlas_type_2015_mining_no",
    "auth_3mth_post_acute_res",
    "auth_3mth_acute_inf",
    "atlas_low_employment_2015_update",
    "auth_3mth_non_er",
    "auth_3mth_acute_cad",
    "cms_orig_reas_entitle_cd",
    "bh_ncal_ind",
    "atlas_type_2015_recreation_no",
    "auth_3mth_post_acute",
    "auth_3mth_facility",
    "atlas_population_loss_2015_update",
    "auth_3mth_home",
    "atlas_farm_to_school13",
    "auth_3mth_acute_inj",
    "auth_3mth_acute",
    "auth_3mth_dc_left_ama",
    "auth_3mth_bh_acute_men",
    "auth_3mth_dc_custodial",
    "auth_3mth_acute_hdz",
    "auth_3mth_rehab",
    "auth_3mth_snf_post_hsp",
    "auth_3mth_dc_home_health",
    "rx_gpi2_56_dist_gpi6_pmpm_ct_3to6m_b4",
    "auth_3mth_acute_cir",
    "atlas_persistentchildpoverty_1980_2011",
    "ccsp_065_pmpm_ct",
    "auth_3mth_post_er",
    "auth_3mth_acute_sns",
    "auth_3mth_dc_other",
    "auth_3mth_bh_acute_mean_los",
    "auth_3mth_acute_mus",
    "atlas_perpov_1980_0711",
    "auth_3mth_acute_gus",
    "atlas_low_education_2015_update",
    "race_cd"
  )


train_raw <- read_csv("2021_Competition_Training.csv", na = c("*", "")) %>%
  clean_names()

test_raw <- read_csv("2021_Competition_Holdout.csv", na = c("*", "")) %>%
  clean_names()

train <- train_raw %>% 
  select(-1, -2, -all_of(zv)) %>% 
  mutate(src_div_id = str_remove(src_div_id, "00"),
         across(.cols = conv, as.character),
         covid_vaccination = ifelse(covid_vaccination == "vacc", 0, 1),
         across(where(is.character), as.factor))

test <- test_raw %>% 
  select(-1, -2, -all_of(zv)) %>% 
  mutate(src_div_id = str_remove(src_div_id, "00"),
         across(.cols = conv, as.character),
         across(where(is.character), as.factor))


y_train <- train$covid_vaccination
x_train <- train %>% select(-covid_vaccination)

train_pool <- catboost.load_pool(data = x_train, label = y_train)
test_pool <- catboost.load_pool(data = test)

fit <- catboost.train(learn_pool = train_pool,
                      params = list(learning_rate = 0.05,
                                    loss_function = "Logloss",
                                    eval_metric = "AUC"))

write_rds("cb_fit_best.rds")


var_imp <- catboost.get_feature_importance(fit)

tibble(var = rownames(var_imp),
       imp = var_imp[,1]) %>%
  slice_max(n = 20, order_by = imp) %>%
  mutate(var = fct_reorder(var, imp)) %>%
  ggplot(aes(x = var, y = imp)) +
  geom_col() +
  coord_flip() +
  theme_light() +
  labs(y = "Importance",
       x = "Variable")

var_shap <- catboost.get_feature_importance(fit, pool = test_pool, type = "ShapValues")

dim(var_shap)
