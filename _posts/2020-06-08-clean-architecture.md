---
layout: post
title: Clean Architecture
featured-img: clean_architecture
category: [architecture]
mathjax: true
summary: Entrando no mundo da produção de conteúdo! Espero que esse início me leve para uma jornada cheia de descobertas.
---

# Clean Architecture: o que é, vantagens e como utilizar em aplicações na prática

## Princípios da arquitetura limpa

### A Regra de Dependência: Software Dependente ?

A Regra de Dependência de acordo com a Figura 1, nos mostra que a seta apenas segue para dentro, ou seja, a camada mais externa visualiza e utiliza a mais interna. O contrário em nenhuma hipótese pode acontecer. Isso deixa o software dependente? Restritivo? Muito pelo contrário, a aplicação e utilização da Dependency Rule, diminui as limitações do código o tornando organizado e acessível. Não ultrapassar essas barreiras é essencial para otimizar o funcionamento do sistema.


### Entidades (Nível mais interno)

Na camada de entidades, deve ser alocada a lógica de negócio e as regras de alto nível. Essa camada pode ser usada por todas as outras, tendo em vista que possui regras mais gerais do software, ou seja, as entidades são usadas pelas classes mais externas.

### Casos de Uso

Neste nível do software, cabe a aplicação das regras de negócio para cada caso. Estas são mais específicas e dizem respeito a validações por exemplo. Como citado por Uncle(2012), esta camada direciona as entidades para utilizar as regras de negócio corporativa para atingir as metas de uso.

### Adaptadores de Interface

Esta camada tem uma função bem específica como todas as outras. Função esta que é converter as informações vindas das camadas internas (entidades+casos de uso) para o reconhecimento dos elementos pertencentes no próximo nível.

### Camadas de estruturas e drivers

Composta pelos elementos mais externos: Frameworks, Banco de dados, bibliotecas e derivações.

## Aplicação prática

As I wanted to use a very expressive model (`xgboost`) and tune the hell out of it (with the TPE algorithm from `hyperopt`), I needed to devise a robust validation process. One of Kaggle greatest competitors, [Lucas Eustaquio](https://www.kaggle.com/leustagos) (who, unfortunately, lost the battle against cancer during the competition), mentioned at an interview that validation is one of the things [he cared the most about](http://blog.kaggle.com/2016/02/22/profiling-top-kagglers-leustagos-current-7-highest-1/), being the first thing that he builds at the start of a competition.

In order to get stable AUC measurements (0.003 of AUC would mean 1,350 positions in the LB) and achieve my goals, I used two CV strategies to evaluate my models:

![]({{ "assets/img/clean_architecture/00.png" | absolute_url }})

In order to get stable AUC measurements (0.003 of AUC would mean 1,350 positions in the LB) and achieve my goals, I used two CV strategies to evaluate my models:

![]({{ "assets/img/clean_architecture/03.png" | absolute_url }})

In order to get stable AUC measurements (0.003 of AUC would mean 1,350 positions in the LB) and achieve my goals, I used two CV strategies to evaluate my models:

![]({{ "assets/img/clean_architecture/03.png" | absolute_url }})

In order to get stable AUC measurements (0.003 of AUC would mean 1,350 positions in the LB) and achieve my goals, I used two CV strategies to evaluate my models:

![]({{ "assets/img/clean_architecture/04.png" | absolute_url }})

In order to get stable AUC measurements (0.003 of AUC would mean 1,350 positions in the LB) and achieve my goals, I used two CV strategies to evaluate my models:

![]({{ "assets/img/clean_architecture/05.png" | absolute_url }})

In order to get stable AUC measurements (0.003 of AUC would mean 1,350 positions in the LB) and achieve my goals, I used two CV strategies to evaluate my models:

![]({{ "assets/img/clean_architecture/06.png" | absolute_url }})

In order to get stable AUC measurements (0.003 of AUC would mean 1,350 positions in the LB) and achieve my goals, I used two CV strategies to evaluate my models:

![]({{ "assets/img/clean_architecture/07.png" | absolute_url }})

In order to get stable AUC measurements (0.003 of AUC would mean 1,350 positions in the LB) and achieve my goals, I used two CV strategies to evaluate my models:

![]({{ "assets/img/clean_architecture/08.png" | absolute_url }})

In order to get stable AUC measurements (0.003 of AUC would mean 1,350 positions in the LB) and achieve my goals, I used two CV strategies to evaluate my models:

![]({{ "assets/img/clean_architecture/final.png" | absolute_url }})

### Considerações Finais
As ideias propostas na obra de Uncle Bob, padronizam o desenvolvimento de software moderno. Observa-se que os métodos sugeridos, se aplicados de maneira correta, organizam o código, facilita o trabalho em equipe, e entre diversos argumentos citados durante esta pesquisa. A implementação do mesmo, não é algo difícil e livrará o desenvolvedor de vários problemas que por ventura pudessem ocorrer durante o projeto de software.

## Referências

Disponível em: https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html

Atensiosamente, 
Thiago

