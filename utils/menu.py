class MenuInterativo:
    def __init__(self, sistema):
        self.sistema = sistema

    def executar(self):
        while True:
            print("\n===== MENU VR/VA =====")
            print("1. Carregar dados")
            print("2. Validar dados")
            print("3. Aplicar regras de negócio")
            print("4. Calcular VR")
            print("5. Gerar relatórios")
            print("6. Executar pipeline completo")
            print("0. Sair")
            opcao = input("Escolha uma opção: ")
            if opcao == '1':
                sucesso = self.sistema._etapa_1_carregar_dados()
                print("Dados carregados!" if sucesso else "Erro ao carregar dados.")
            elif opcao == '2':
                sucesso = self.sistema._etapa_2_validar_dados()
                print("Dados validados!" if sucesso else "Erro na validação.")
            elif opcao == '3':
                sucesso = self.sistema._etapa_3_aplicar_regras()
                print("Regras aplicadas!" if sucesso else "Erro ao aplicar regras.")
            elif opcao == '4':
                sucesso = self.sistema._etapa_4_calcular_vr()
                print("Cálculo realizado!" if sucesso else "Erro no cálculo.")
            elif opcao == '5':
                sucesso = self.sistema._etapa_5_gerar_relatorios()
                print("Relatórios gerados!" if sucesso else "Erro ao gerar relatórios.")
            elif opcao == '6':
                sucesso = self.sistema.executar()
                print("Pipeline completo executado!" if sucesso else "Erro no pipeline.")
            elif opcao == '0':
                print("Saindo do sistema...")
                break
            else:
                print("Opção inválida. Tente novamente.")
