---
layout: post
title: Clean Architecture
featured-img: clean_architecture
category: [architecture]
mathjax: true
summary: Vamos falar um pouco sobre essa "nova" forma de estruturar seus projetos, suas vantagens e desvantagens, e um exemplo prático de sua utilização 
---

# Clean Architecture: o que é, vantagens e como utilizar em aplicações na prática

Ultimamente vemos ouvindo falar bastante sobre termos como "Clean Architecture", "Hexagonal Architecture", "SOLID", "DRY", "DDD", etc. Todos eles são princípios, práticas ou abordagens no mundo de desenvolvimento de software que, em essência, convergem para o objetivo de tornar o software mais robusto, escalável e flexível, deixando-o tolerante a mudanças, facilitando a implementação de novos requisitos para a evolução e manutenção do sistema.

Quando se fala em Clean archi temos:

- Arquitetura FOCADA nas regras de negócio: suas regras não devem conhecer o "mundo lá fora" (Frameworks, Ferramentas, Tecnologias, etc).
- Inversão de dependências: Por exemplo, seu banco de dados deve depender das suas regras, e não suas regras do seu banco. Sua UI? Mesma coisa!
- A melhor decisão é a que você pode tomar depois?: Isso não é preguiça, é arquitetura incremental. 
- Regras de negócio devem ser 100% TESTÁVEIS e INDEPENDENTES!

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

API rota de cadastro

Utiliza o Express, MongoDB, BCrypt e Validator para validar o e-mail recebido

Geralmente como as pessoas fazem.

![]({{ "assets/img/clean_architecture/01.png" | absolute_url }})

Nesse cenário temos componentes como Controller e Router aclopados a biblioteca de terceiros, além de termos componente que tem muitas "responsabilidades". Se precisar em algum momento trocar o Express por outro framework, será necessário alterar todos os controllers

Então para dar início a uma transformação na arquitetura do nosso sistema podemos iniciar focando em desacoplar nossos controllers do Express. Esse tipo de decisão em um cenário ideal, deveria ser uma tarefa muito fácil. Mas aqui temos que essa biblioteca praticamente toma conta do sistema.

Hoje o `SignUpController` aponta para o Express por isso precisamos usar o padrão de inversão de dependência (dependency inversion) para fazer com que o Express olhe para nossos controllers.

![]({{ "assets/img/clean_architecture/03.png" | absolute_url }})

Para isso, podemos criar um adapter entre os dois que terá a tarefa de converter as interfaces do controller para a realidade do Express. Uma das particularidades do Express é que ele espera receber o (req, res) como parâmetros nas rotas definidas. 

![]({{ "assets/img/clean_architecture/04.png" | absolute_url }})

Também não podemos permitir que o adapter dependa diretamente do `SignUpController`. Devemos fazer com que ele possa adaptar qualquer controlador, assim, criamos uma interface `Controller`, que irá servir como um limite da camada de apresentação para fazer a inversão de dependência
O adapter precisa de qualquer classe que utilize a interface.
Com isso a dependência inverteu. Se eu precisar trocar, só altero o adapter

![]({{ "assets/img/clean_architecture/05.png" | absolute_url }})

Seguindo em frente, precisamos agora desacoplar biblioteca para validar e-mail para que nosso presentation layer não dependa de um componente externo. Criamos um `EmailValidatorAdapter` semelhante ao adapter criado anteriormente.
Assim, em vez de o `SignUpController` depender diretamente desse adapter, definimos ainda na camada de apresentação, uma nova interface `EmailValidator` que diz o que esse componente deve fazer e "alguém" de fora implementa a interface definida. Dessa forma desacoplamos e se outros controladores precisarem usar esse validator poderemos facilmente reutilizar o componente. Além do benefício de, se um dia optarmos por substituir essa biblioteca por uma regex por exemplo, alteraremos apenas um componente.
Podemos dizer que é nossa camada **Utils** que é mais genérica irá conter coisas que podem ser utilizadas em qualquer lugar.

![]({{ "assets/img/clean_architecture/06.png" | absolute_url }})

Surge, então, a necessidade de uma camada de negócio que diga o que precisamos fazer. Pois para realizar um cadastro, precisamos salvar os dados no banco de dados, mas antes disso, queremos criptografar a senha do usuário.
Criamos então uma inteface `AddAccount` que nada mais é que a representação de uma camada de negócio da aplicação Camada **Domain**. Essa camada não terá implementação, apenas protocolos que dizem o que nossa regra de negócio deve fazer.
Com isso, o `SignUpController` irá precisar de alguém que implemente essa interface para criar uma conta de usuário. Não importa se a implementação será com banco de dados, cache, ou dados mockados, o que importa é que a implementação respeite a interface definida.

Assumindo que queremos a implementação da regra de negócio voltada para armazenamento em banco de dados, criaremos a **Data** layer que terá o componente `DbAddAcccount`.
Esse sim irá utilizar o BCrypt para fazer a criptografia da senha do usuário. Mas não queremos acopla-los diretamente. Assim como antes, criaremos um adapter para isolar os componentes.
Ele ficará dentro da **Infra** layer, camada que terá implementações de interface voltadas para frameworks. E agora para realizar a inversão de dependência, criamos a interface `Encrypter` ainda na Data layer já que o `DbAddAcccount` precisa de alguém que saiba fazer criptografia e não ele mesmo saber como faz.
Com isso a dependência inverteu novamente. O Infra layer que aponta para o Data layer e não o Data layer apontando para o Infra.

![]({{ "assets/img/clean_architecture/07.png" | absolute_url }})

Para finalizar temos o a dependência com o MongoDb. Criaremos o componente `AddAccountMongoRepository` que sabe usar o mongo e criaremos também a interface `AddUserRepo` pois outros componentes podem precisar de "alguém" que saiba inserir no banco de dados, não importa qual.
Inversão de dependência feita mais uma vez.

Com isso as bibliotecas de  terceiros estão cada vez mais isoladas, cada vez mais na camada mais "fora" da nossa arquitetura. Se futuramente quisermos trocar MongoDB, vamos alterar apenas um componente.

![]({{ "assets/img/clean_architecture/08.png" | absolute_url }})

Realizamos até aqui transformações, com a ajuda do padrão de projeto *Adapter*, que removem o acoplamento entre as camadas. Mas para conseguir ter uma solução completa e desacoplar nossas camadas precisamos acoplar uma delas, e esta será a **Main** layer. Ela será responsável por criar instancias de todos os objetos.
Exemplo: Para criar a rota de signup precisamos do `SignUpController`, que por sua vez precisa de alguém que implemente a interface `AddAccount`. Mas que sua implementação não será instanciada no controller.
Alguém irá criar essa instancia e injetará no controller. O Main layer fará a composição desses objetos através de outro design pattern, o *Composite*, e toda composição será feita em um lugar só.

Abaixo o desenho final desse exemplo. 
Destacando com cores para melhor visualização. Dependências sempre nas "pontas". Facilmente podemos trocar sem afetar o resto do sistema.

![]({{ "assets/img/clean_architecture/final.png" | absolute_url }})

### Considerações Finais
As ideias propostas na obra de Uncle Bob, padronizam o desenvolvimento de software moderno. Observa-se que os métodos sugeridos, se aplicados de maneira correta, organizam o código, facilita o trabalho em equipe, e entre diversos argumentos citados durante esta pesquisa. A implementação do mesmo, não é algo difícil e livrará o desenvolvedor de vários problemas que por ventura pudessem ocorrer durante o projeto de software.

## Referências

Artigo do [Uncle Bob](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) sobre Clean Architecture.

Curso NodeJs, Typescript, TDD, Clean Architecture e SOLID na [Udemy](https://www.udemy.com/course/tdd-com-mango/) do Rodrigo Manguinho.

Atensiosamente, 
Thiago
